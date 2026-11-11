//! Display-significant spreadsheet formats. Never evaluates formulas.
use super::*;

#[derive(Clone, Debug)]
enum Token {
    Pattern(char),
    Literal(String),
}

fn tokens(code: &str) -> Result<Vec<Token>> {
    if code.len() > 1024 {
        return Err(XbergError::validation("Excel number format exceeds 1024 bytes"));
    }
    let mut out = Vec::new();
    let mut chars = code.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '"' => {
                let mut literal = String::new();
                let mut closed = false;
                for next in chars.by_ref() {
                    if next == '"' {
                        closed = true;
                        break;
                    }
                    literal.push(next);
                }
                if !closed {
                    return Err(XbergError::parsing("Unterminated Excel format literal"));
                }
                out.push(Token::Literal(literal));
            }
            '\\' => {
                if let Some(next) = chars.next() {
                    out.push(Token::Literal(next.to_string()));
                }
            }
            '_' | '*' => {
                chars.next();
            }
            '[' => {
                let mut directive = String::new();
                let mut closed = false;
                for next in chars.by_ref() {
                    if next == ']' {
                        closed = true;
                        break;
                    }
                    directive.push(next);
                }
                if !closed {
                    return Err(XbergError::parsing("Unterminated Excel format directive"));
                }
                if let Some(currency) = directive.strip_prefix('$') {
                    let symbol = currency.split('-').next().unwrap_or("");
                    if !symbol.is_empty() {
                        out.push(Token::Literal(symbol.to_owned()));
                    }
                } else if directive
                    .chars()
                    .all(|c| matches!(c, 'h' | 'H' | 'm' | 'M' | 's' | 'S'))
                {
                    out.push(Token::Literal(format!("[{directive}]")));
                }
            }
            _ => out.push(Token::Pattern(ch)),
        }
    }
    Ok(out)
}

fn sections(code: &str) -> Vec<&str> {
    let mut result = Vec::new();
    let mut quoted = false;
    let mut escaped = false;
    let mut start = 0;
    for (i, ch) in code.char_indices() {
        if ch == ';' && !quoted && !escaped {
            result.push(&code[start..i]);
            start = i + 1;
        }
        if ch == '"' && !escaped {
            quoted = !quoted;
        }
        escaped = ch == '\\' && !escaped;
    }
    result.push(&code[start..]);
    result
}

fn select_section(value: f64, code: &str) -> (&str, bool) {
    let parts = sections(code);
    static CONDITION: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(r"\[(<=|>=|<>|=|<|>)([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\]").expect("static regex")
    });
    let condition = &*CONDITION;
    if parts.iter().take(2).any(|p| condition.is_match(p)) {
        for part in parts.iter().take(3) {
            if let Some(capture) = condition.captures(part) {
                if let Ok(other) = capture[2].parse::<f64>() {
                    let yes = match &capture[1] {
                        "<=" => value <= other,
                        ">=" => value >= other,
                        "<>" => value != other,
                        "=" => value == other,
                        "<" => value < other,
                        _ => value > other,
                    };
                    if yes {
                        return (part, false);
                    }
                }
            } else {
                return (part, false);
            }
        }
        return ("General", false);
    }
    let index = if value == 0.0 && parts.len() > 2 {
        2
    } else if value < 0.0 && parts.len() > 1 {
        1
    } else {
        0
    };
    (parts[index], index == 1)
}

/// Decimal half-up rounding from the stored decimal representation; avoids binary
/// floating point ties such as 1.005 becoming 1.00. Input digits are bounded by f64.
fn fixed(value: f64, shift: i32, precision: usize) -> Result<String> {
    let raw = value.abs().to_string();
    let (mantissa, exponent) = raw
        .split_once(['e', 'E'])
        .map_or((raw.as_str(), 0), |(a, b)| (a, b.parse::<i32>().unwrap_or(0)));
    let decimal = mantissa.split_once('.').map_or(0, |(_, b)| b.len() as i32);
    let digits: String = mantissa.chars().filter(|&c| c != '.').collect();
    let coefficient = digits
        .parse::<u128>()
        .map_err(|_| XbergError::validation("Excel number exceeds decimal display capacity"))?;
    let power = exponent - decimal + shift + precision as i32;
    let rounded = if power >= 0 {
        coefficient.checked_mul(
            10u128
                .checked_pow(power as u32)
                .ok_or_else(|| XbergError::validation("Excel format magnitude is too large"))?,
        )
    } else if power < -38 {
        Some(0)
    } else {
        let divisor = 10u128.pow((-power) as u32);
        coefficient.checked_add(divisor / 2).map(|n| n / divisor)
    }
    .ok_or_else(|| XbergError::validation("Excel format magnitude is too large"))?;
    let digits = format!("{rounded:0width$}", width = precision + 1);
    let cut = digits.len() - precision;
    let mut rendered = digits[..cut].to_owned();
    if precision > 0 {
        rendered.push('.');
        rendered.push_str(&digits[cut..]);
    }
    Ok(rendered)
}

pub(super) fn number(value: f64, code: &str) -> Result<String> {
    if code.len() > 1024 {
        return Err(XbergError::validation("Excel number format exceeds 1024 bytes"));
    }
    if !value.is_finite() {
        return Err(XbergError::validation("Excel contains a non-finite number"));
    }
    let (section, negative_section) = select_section(value, code);
    let ts = tokens(section)?;
    let body: String = ts
        .iter()
        .filter_map(|t| if let Token::Pattern(c) = t { Some(*c) } else { None })
        .collect();
    if body.to_ascii_lowercase().contains("general") || body.contains('@') {
        return Ok(format_excel_float(value));
    }
    let slots: Vec<usize> = ts
        .iter()
        .enumerate()
        .filter_map(|(i, t)| matches!(t, Token::Pattern('0' | '#' | '?')).then_some(i))
        .collect();
    if slots.is_empty() {
        return Ok(ts
            .iter()
            .map(|t| match t {
                Token::Literal(s) => s.clone(),
                Token::Pattern(c) => c.to_string(),
            })
            .collect());
    }
    // Identifier masks keep literal separators at their original position.
    if value.fract() == 0.0
        && body.matches('0').count() > 1
        && !body
            .chars()
            .any(|c| matches!(c, '.' | '%' | '/' | '#' | '?' | 'e' | 'E' | ','))
    {
        let digits = format!("{:0width$.0}", value.abs(), width = slots.len());
        let mut remaining = digits.as_str();
        let mut result = String::new();
        for t in ts {
            match t {
                Token::Pattern('0') => {
                    let n = if remaining.len() > slots.len() {
                        remaining.len() - slots.len() + 1
                    } else {
                        1
                    };
                    result.push_str(&remaining[..n.min(remaining.len())]);
                    remaining = &remaining[n.min(remaining.len())..];
                }
                Token::Pattern(c) => result.push(c),
                Token::Literal(s) => result.push_str(&s),
            }
        }
        return Ok(format!(
            "{}{result}",
            if value < 0.0 && !negative_section { "-" } else { "" }
        ));
    }
    if body.contains('/') {
        let scaled = value * 100f64.powi(body.matches('%').count() as i32);
        if !scaled.is_finite() {
            return Err(XbergError::validation("Excel fraction scale exceeds numeric capacity"));
        }
        let rendered = fraction(if negative_section { scaled.abs() } else { scaled }, &body)?;
        let slash = ts
            .iter()
            .position(|t| matches!(t, Token::Pattern('/')))
            .expect("fraction slash");
        let end = ts
            .iter()
            .enumerate()
            .skip(slash + 1)
            .take_while(|(_, t)| matches!(t,Token::Pattern(c) if c.is_ascii_digit() || matches!(c,'#'|'?')))
            .map(|(i, _)| i)
            .last()
            .unwrap_or(slash);
        let literal = |ts: &[Token]| -> String {
            ts.iter()
                .map(|t| match t {
                    Token::Literal(s) => s.clone(),
                    Token::Pattern(c) => c.to_string(),
                })
                .collect()
        };
        return Ok(format!(
            "{}{rendered}{}",
            literal(&ts[..slots[0]]),
            literal(&ts[end + 1..])
        ));
    }
    let precision_code = body.split_once('.').map_or("", |(_, tail)| {
        tail.split(|c: char| !matches!(c, '0' | '#' | '?')).next().unwrap_or("")
    });
    let precision = precision_code.len();
    if precision > 30 {
        return Err(XbergError::validation("Excel format precision exceeds 30"));
    }
    let percents = body.matches('%').count();
    let last_slot = *slots.last().expect("non-empty");
    let scale = ts[last_slot + 1..]
        .iter()
        .take_while(|t| matches!(t, Token::Pattern(',')))
        .count();
    let mut rendered = if body.contains(['E', 'e']) {
        let scientific = format!(
            "{:.*E}",
            precision,
            value.abs() * 100f64.powi(percents as i32) / 1000f64.powi(scale as i32)
        );
        let (m, e) = scientific.split_once('E').expect("scientific format");
        let exponent = e.parse::<i32>().unwrap_or(0);
        let exp_width = body
            .split(['E', 'e'])
            .nth(1)
            .unwrap_or("")
            .chars()
            .filter(|&c| c == '0')
            .count();
        format!(
            "{m}E{}{:0width$}",
            if exponent < 0 { "-" } else { "+" },
            exponent.unsigned_abs(),
            width = exp_width
        )
    } else {
        fixed(value, 2 * percents as i32 - 3 * scale as i32, precision)?
    };
    if !body.contains(['E', 'e']) {
        if let Some((integer, decimal)) = rendered.split_once('.') {
            let minimum = precision_code.trim_end_matches(['#', '?']).len();
            let decimal = decimal.trim_end_matches('0');
            rendered = if decimal.len() < minimum {
                format!("{integer}.{decimal:0<minimum$}")
            } else if decimal.is_empty() {
                integer.to_owned()
            } else {
                format!("{integer}.{decimal}")
            };
        }
    }
    if !body.contains(['E', 'e']) {
        let minimum = body
            .split('.')
            .next()
            .unwrap_or("")
            .chars()
            .filter(|&c| c == '0')
            .count();
        let (integer, tail) = rendered
            .split_once('.')
            .map_or((rendered.as_str(), None), |(a, b)| (a, Some(b)));
        if integer.len() < minimum {
            let padded = format!("{integer:0>minimum$}");
            rendered = tail.map_or_else(|| padded.clone(), |tail| format!("{padded}.{tail}"));
        }
    }
    if value < 0.0 && !negative_section {
        rendered.insert(0, '-');
    }
    rendered.push_str(&"%".repeat(percents));
    let first = slots[0];
    let affix = |slice: &[Token]| -> String {
        slice
            .iter()
            .map(|t| match t {
                Token::Literal(s) => s.clone(),
                Token::Pattern(c) if !matches!(c, ',' | '%' | '0' | '#' | '?' | 'E' | 'e' | '.') => c.to_string(),
                _ => String::new(),
            })
            .collect()
    };
    let prefix = affix(&ts[..first]);
    let suffix = affix(&ts[last_slot + 1..]);
    Ok(format!("{prefix}{rendered}{suffix}"))
}

fn fraction(value: f64, body: &str) -> Result<String> {
    let (left, right) = body.split_once('/').expect("fraction");
    let denominator_code: String = right
        .trim()
        .chars()
        .take_while(|c| c.is_ascii_digit() || matches!(c, '#' | '?'))
        .collect();
    let fixed_denominator = denominator_code.parse::<u32>().ok().filter(|&n| n > 0);
    let limit = fixed_denominator
        .unwrap_or_else(|| 10u32.pow(denominator_code.len().min(4) as u32) - 1)
        .max(1);
    let mut best = (0u64, 1u32, f64::INFINITY);
    let candidates = if fixed_denominator.is_some() {
        limit..=limit
    } else {
        1..=limit
    };
    for d in candidates {
        let n = (value.abs() * f64::from(d)).round() as u64;
        let error = (value.abs() - n as f64 / f64::from(d)).abs();
        if error < best.2 {
            best = (n, d, error);
        }
    }
    let (mut n, d, _) = best;
    let whole = if left.contains(' ') {
        let w = n / u64::from(d);
        n %= u64::from(d);
        w
    } else {
        0
    };
    let text = if whole > 0 && n > 0 {
        format!("{whole} {n}/{d}")
    } else if whole > 0 {
        whole.to_string()
    } else {
        format!("{n}/{d}")
    };
    Ok(format!("{}{text}", if value < 0.0 { "-" } else { "" }))
}

pub(super) fn apply(range: &mut Range<Data>, formats: &calamine::StyleRange) -> Result<()> {
    let positions: Vec<_> = range.used_cells().map(|(r, c, _)| (r, c)).collect();
    let (row_start, col_start) = range.start().unwrap_or((0, 0));
    for (row, col) in positions {
        let pos = (row_start + row as u32, col_start + col as u32);
        let Some(code) = formats
            .get(pos)
            .and_then(|s| s.number_format.as_ref())
            .map(|f| f.format_code.as_str())
        else {
            continue;
        };
        let formatted = match range.get_value(pos) {
            Some(Data::Float(n)) => Some(number(*n, code)?),
            Some(Data::Int(n)) => Some(number(*n as f64, code)?),
            Some(Data::DateTime(dt)) => Some(temporal(dt, code)?),
            _ => None,
        };
        if let Some(value) = formatted {
            range.set_value(pos, Data::String(value));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn display_significant_numbers() {
        for (value, format, expected) in [
            (0.001, "0.0%", "0.1%"),
            (0.1575, "0.00%", "15.75%"),
            (15.0, "0\"%\"", "15%"),
            (1234.5, "#,##0.00 \"₽\"", "1234.50 ₽"),
            (42.0, "000000", "000042"),
            (1.005, "0.00", "1.01"),
            (1234567.0, "0.0,,", "1.2"),
            (0.25, "# ?/?", "1/4"),
            (-42.0, "0;(0)", "(42)"),
            (1200.0, "0.00E+00", "1.20E+03"),
        ] {
            assert_eq!(number(value, format).unwrap(), expected, "{value} {format}");
        }
    }
}

pub(super) fn temporal(dt: &calamine::ExcelDateTime, code: &str) -> Result<String> {
    if dt.is_duration() {
        return Ok(super::datetime::format_excel_datetime(dt));
    }
    let (year, month, day, hour, minute, second, millis) = dt.to_ymd_hms_milli();
    let ts = tokens(sections(code)[0])?;
    let mut out = String::new();
    let mut index = 0usize;
    let has_ampm = code.to_ascii_uppercase().contains("AM/PM");
    let months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    while index < ts.len() {
        if let Token::Literal(text) = &ts[index] {
            out.push_str(text);
            index += 1;
            continue;
        }
        let Token::Pattern(ch) = ts[index] else { unreachable!() };
        if has_ampm && ch.eq_ignore_ascii_case(&'a') && index + 4 < ts.len() {
            out.push_str(if hour < 12 { "AM" } else { "PM" });
            index += 5;
            continue;
        }
        let lower = ch.to_ascii_lowercase();
        let mut length = 1;
        while matches!(ts.get(index+length),Some(Token::Pattern(next)) if next.to_ascii_lowercase()==lower) {
            length += 1;
        }
        let previous = ts[..index].iter().rev().find_map(|t| match t {
            Token::Pattern(c) if c.is_ascii_alphabetic() => Some(c.to_ascii_lowercase()),
            _ => None,
        });
        let next = ts[index + length..].iter().find_map(|t| match t {
            Token::Pattern(c) if c.is_ascii_alphabetic() => Some(c.to_ascii_lowercase()),
            _ => None,
        });
        let is_minute = lower == 'm' && (previous == Some('h') || next == Some('s'));
        let value = match lower {
            '0' if index > 0 && matches!(ts.get(index - 1), Some(Token::Pattern('.'))) => {
                let fraction = format!("{millis:03}");
                if length <= 3 {
                    fraction[..length].to_owned()
                } else {
                    format!("{fraction}{}", "0".repeat(length - 3))
                }
            }
            'y' => {
                if length <= 2 {
                    format!("{:02}", year % 100)
                } else {
                    format!("{year:04}")
                }
            }
            'm' if !is_minute && length >= 3 => {
                let name = months.get(month.saturating_sub(1) as usize).unwrap_or(&"");
                if length == 3 {
                    name[..name.len().min(3)].to_owned()
                } else {
                    name.to_string()
                }
            }
            'm' if is_minute => {
                if length == 1 {
                    minute.to_string()
                } else {
                    format!("{minute:02}")
                }
            }
            'm' => {
                if length == 1 {
                    month.to_string()
                } else {
                    format!("{month:02}")
                }
            }
            'd' => {
                if length == 1 {
                    day.to_string()
                } else if length == 2 {
                    format!("{day:02}")
                } else {
                    let date = chrono::NaiveDate::from_ymd_opt(year as i32, month as u32, day as u32)
                        .ok_or_else(|| XbergError::parsing("Invalid Excel calendar date"))?;
                    date.format(if length == 3 { "%a" } else { "%A" }).to_string()
                }
            }
            'h' => {
                let hour = if has_ampm {
                    if hour % 12 == 0 { 12 } else { hour % 12 }
                } else {
                    hour
                };
                if length == 1 {
                    hour.to_string()
                } else {
                    format!("{hour:02}")
                }
            }
            's' => {
                if length == 1 {
                    second.to_string()
                } else {
                    format!("{second:02}")
                }
            }
            _ => ch.to_string().repeat(length),
        };
        out.push_str(&value);
        index += length;
    }
    Ok(out)
}
