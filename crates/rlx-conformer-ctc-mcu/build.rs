// The model is too large to commit; it is linked in from the exporter's
// output. Override with RCC_MODEL=/path/to/model.rcc.
fn main() {
    let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let model = std::env::var("RCC_MODEL")
        .unwrap_or_else(|_| format!("{dir}/../../.cache/conformer-ctc-mcu/small-int8.rcc"));
    println!("cargo:rustc-env=RCC_MODEL={model}");
    println!("cargo:rerun-if-env-changed=RCC_MODEL");
    println!("cargo:rerun-if-changed={model}");
    println!("cargo:rerun-if-changed=link.x");
    println!("cargo:rustc-link-search={dir}");
}
