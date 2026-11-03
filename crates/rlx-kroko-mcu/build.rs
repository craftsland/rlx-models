// The model is too large to commit; it is linked in from the exporter's
// output. Override with RKR_MODEL=/path/to/model.rkr, and point RKR_DATA at
// the matching `mcu_vectors` output (default `data/`, Kroko-L int4).
fn main() {
    let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let model = std::env::var("RKR_MODEL")
        .unwrap_or_else(|_| format!("{dir}/../../.cache/kroko-mcu/en64-int4.rkr"));
    println!("cargo:rustc-env=RKR_MODEL={model}");
    println!("cargo:rerun-if-env-changed=RKR_MODEL");
    println!("cargo:rerun-if-changed={model}");
    let data = std::env::var("RKR_DATA").unwrap_or_else(|_| format!("{dir}/data"));
    println!("cargo:rustc-env=RKR_DATA={data}");
    println!("cargo:rerun-if-env-changed=RKR_DATA");
    for f in ["pcm_i16.bin", "ids_u16.bin", "encoder_fnv.txt"] {
        println!("cargo:rerun-if-changed={data}/{f}");
    }
    println!("cargo:rerun-if-changed=link.x");
    println!("cargo:rustc-link-search={dir}");
}
