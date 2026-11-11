//! BIFF display formats not currently exposed by calamine's XLS style API.
//! Values and formulas still come from calamine; this bounded pass reads XF indices.
use super::*;
pub(super) type SheetFormats = HashMap<(u32, u32), String>;

fn word(data: &[u8], at: usize) -> Result<u16> {
    data.get(at..at + 2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .ok_or_else(|| XbergError::parsing("Truncated XLS style record"))
}
fn builtin(id: u16) -> Option<&'static str> {
    Some(match id {
        0 => "General",
        1 => "0",
        2 => "0.00",
        3 => "#,##0",
        4 => "#,##0.00",
        5 | 6 => "$#,##0;($#,##0)",
        7 | 8 => "$#,##0.00;($#,##0.00)",
        9 => "0%",
        10 => "0.00%",
        11 => "0.00E+00",
        12 => "# ?/?",
        13 => "# ??/??",
        14 => "mm-dd-yy",
        15 => "d-mmm-yy",
        16 => "d-mmm",
        17 => "mmm-yy",
        18 => "h:mm AM/PM",
        19 => "h:mm:ss AM/PM",
        20 => "h:mm",
        21 => "h:mm:ss",
        22 => "m/d/yy h:mm",
        37 | 38 => "#,##0;(#,##0)",
        39 | 40 => "#,##0.00;(#,##0.00)",
        45 => "mm:ss",
        46 => "[h]:mm:ss",
        47 => "mmss.0",
        48 => "##0.0E+0",
        49 => "@",
        _ => return None,
    })
}
fn record(data: &[u8], position: &mut usize) -> Result<Option<(u16, Vec<u8>)>> {
    if *position == data.len() {
        return Ok(None);
    }
    let kind = word(data, *position)?;
    let length = usize::from(word(data, *position + 2)?);
    let start = *position + 4;
    let end = start
        .checked_add(length)
        .ok_or_else(|| XbergError::parsing("XLS record overflow"))?;
    let payload = data
        .get(start..end)
        .ok_or_else(|| XbergError::parsing("Truncated XLS style stream"))?
        .to_vec();
    *position = end;
    Ok(Some((kind, payload)))
}
pub(super) fn read(data: &[u8], limits: &SecurityLimits) -> Result<Vec<SheetFormats>> {
    let mut compound = cfb::OpenOptions::new()
        .open_with(Cursor::new(data))
        .map_err(|e| XbergError::parsing(format!("Invalid XLS container: {e}")))?;
    let name = if compound.is_stream("/Workbook") {
        "/Workbook"
    } else {
        "/Book"
    };
    let mut stream = Vec::new();
    compound
        .open_stream(name)?
        .take(limits.max_archive_size as u64 + 1)
        .read_to_end(&mut stream)?;
    if stream.len() > limits.max_archive_size {
        return Err(XbergError::validation("XLS styles exceed archive limit"));
    }
    let mut custom = HashMap::new();
    let mut xfs = Vec::new();
    let mut offsets = Vec::new();
    let mut pos = 0;
    while let Some((kind, payload)) = record(&stream, &mut pos)? {
        match kind {
            0x000A => break,
            0x00E0 => xfs.push(word(&payload, 2)?),
            0x0085 => {
                let b = payload
                    .get(..4)
                    .ok_or_else(|| XbergError::parsing("Truncated BOUNDSHEET"))?;
                offsets.push(u32::from_le_bytes(b.try_into().expect("four bytes")) as usize);
            }
            0x041E => {
                let id = word(&payload, 0)?;
                let length = usize::from(word(&payload, 2)?);
                let flags = *payload
                    .get(4)
                    .ok_or_else(|| XbergError::parsing("Truncated XLS format string"))?;
                let text = if flags & 1 != 0 {
                    let bytes = payload
                        .get(5..5 + length * 2)
                        .ok_or_else(|| XbergError::parsing("Truncated UTF-16 XLS format"))?;
                    String::from_utf16(
                        &bytes
                            .chunks_exact(2)
                            .map(|c| u16::from_le_bytes([c[0], c[1]]))
                            .collect::<Vec<_>>(),
                    )
                    .map_err(|e| XbergError::parsing(e.to_string()))?
                } else {
                    payload
                        .get(5..5 + length)
                        .ok_or_else(|| XbergError::parsing("Truncated XLS format"))?
                        .iter()
                        .map(|&b| char::from(b))
                        .collect()
                };
                custom.insert(id, text);
            }
            _ => {}
        }
    }
    let mut result = Vec::with_capacity(offsets.len());
    for mut pos in offsets {
        let mut cells = SheetFormats::new();
        while let Some((kind, payload)) = record(&stream, &mut pos)? {
            if kind == 0x000A {
                break;
            }
            let mut insert = |row: u32, col: u32, xf: u16| -> Result<()> {
                let id = *xfs
                    .get(usize::from(xf))
                    .ok_or_else(|| XbergError::parsing("XLS cell references unknown XF style"))?;
                if let Some(code) = custom.get(&id).map(String::as_str).or_else(|| builtin(id)) {
                    if code != "General" {
                        cells.insert((row, col), code.to_owned());
                    }
                }
                Ok(())
            };
            match kind {
                0x0203 | 0x027E | 0x0006 => insert(
                    u32::from(word(&payload, 0)?),
                    u32::from(word(&payload, 2)?),
                    word(&payload, 4)?,
                )?,
                0x00BD => {
                    let row = u32::from(word(&payload, 0)?);
                    let col = u32::from(word(&payload, 2)?);
                    let entries = payload.len().saturating_sub(6) / 6;
                    for i in 0..entries {
                        insert(row, col + i as u32, word(&payload, 4 + i * 6)?)?;
                    }
                }
                _ => {}
            }
            if cells.len() > limits.max_table_cells {
                return Err(XbergError::validation("XLS display styles exceed table cell budget"));
            }
        }
        result.push(cells);
    }
    Ok(result)
}
pub(super) fn apply(range: &mut Range<Data>, formats: &SheetFormats) -> Result<()> {
    for (&position, code) in formats {
        let formatted = match range.get_value(position) {
            Some(Data::Float(n)) => Some(super::display::number(*n, code)?),
            Some(Data::Int(n)) => Some(super::display::number(*n as f64, code)?),
            Some(Data::DateTime(dt)) => Some(super::display::temporal(dt, code)?),
            _ => None,
        };
        if let Some(value) = formatted {
            range.set_value(position, Data::String(value));
        }
    }
    Ok(())
}
