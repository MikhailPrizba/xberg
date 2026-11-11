// Copyright 2021, pdfium-sys Developers
// Copyright 2022 - 2024, pdfium-render Developers
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be

fn main() {
    if std::env::var("PDFIUM_STATIC_LIB_PATH").is_ok() {
        println!("cargo:rustc-cfg=pdfium_use_static");
        statically_link_pdfium();
    } else if let Ok(path) = std::env::var("PDFIUM_DYNAMIC_LIB_PATH") {
        println!("cargo:rustc-link-lib=dylib=pdfium");
        println!("cargo:rustc-link-search=native={}", path);
    } else if std::env::var("TARGET")
        .map(|t| t != "wasm32-unknown-unknown")
        .unwrap_or(true)
    {
        // Neither env var is set, so this build links nothing: the crate falls back to its
        // `dynamic_bindings` module, which resolves `FPDF_*` symbols via `libloading::Library`
        // at first Pdfium use, not at link time. That call dlopen()s the platform library name
        // (`libpdfium.so`/`.dylib`/`pdfium.dll`) from the current working directory and then the
        // system library search path -- neither of which this build places anything into. If no
        // caller supplies a Pdfium binary out of band, `Pdfium::default()`/`bind_to_system_library()`
        // return (or, via `Pdfium::default()`, panic on) a load-library error the first time Pdfium
        // extraction actually runs, not during this build. See issue #702.
        println!(
            "cargo:warning=xberg-pdfium-render: no PDFIUM_STATIC_LIB_PATH or PDFIUM_DYNAMIC_LIB_PATH \
             set; this build will compile but cannot load libpdfium at runtime unless one is present \
             on the system library search path or working directory (see issue #702)"
        );
    }

    println!("cargo:rerun-if-env-changed=PDFIUM_STATIC_LIB_PATH");
    println!("cargo:rerun-if-env-changed=PDFIUM_DYNAMIC_LIB_PATH");
}

fn statically_link_pdfium() {
    if let Ok(path) = std::env::var("PDFIUM_STATIC_LIB_PATH") {
        println!("cargo:rustc-link-lib=static=pdfium");
        println!("cargo:rustc-link-search=native={}", path);

        let target = std::env::var("TARGET").unwrap_or_default();
        if target.contains("apple") {
            println!("cargo:rustc-link-lib=dylib=c++");
            println!("cargo:rustc-link-lib=framework=CoreGraphics");
        } else if target.contains("linux") || target.contains("gnu") || target.contains("musl") {
            if target.contains("musl") {
                println!("cargo:rustc-link-lib=static=stdc++");
            } else {
                println!("cargo:rustc-link-lib=dylib=stdc++");
            }
        }
    }
}
