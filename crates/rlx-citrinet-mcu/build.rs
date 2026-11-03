// The model is too large to commit; it is linked in from the exporter's
// output. Override with RCT_MODEL=/path/to/model.rct.
fn main() {
    let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let model = std::env::var("RCT_MODEL")
        .unwrap_or_else(|_| format!("{dir}/../../.cache/citrinet-mcu/c256-int8.rct"));
    println!("cargo:rustc-env=RCT_MODEL={model}");
    println!("cargo:rerun-if-env-changed=RCT_MODEL");
    println!("cargo:rerun-if-changed={model}");
    println!("cargo:rerun-if-changed=link.x");
    println!("cargo:rustc-link-search={dir}");
}
