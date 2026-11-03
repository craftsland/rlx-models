// The model is too large to commit; it is linked in from the exporter's
// output. Override with RMS_MODEL=/path/to/model.rms.
fn main() {
    let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let model = std::env::var("RMS_MODEL")
        .unwrap_or_else(|_| format!("{dir}/../../.cache/moonshine-mcu/tiny.rms"));
    println!("cargo:rustc-env=RMS_MODEL={model}");
    println!("cargo:rerun-if-env-changed=RMS_MODEL");
    println!("cargo:rerun-if-changed={model}");
    println!("cargo:rerun-if-changed=link.x");
    println!("cargo:rustc-link-search={dir}");
}
