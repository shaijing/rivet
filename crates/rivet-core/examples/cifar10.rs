use std::path::PathBuf;
use rivet_core::dataset::{ArrowImageDataset, Dataset};

fn main() {
    let dataset = ArrowImageDataset::new(
        vec![PathBuf::from("/home/ling/.cache/huggingface/datasets/uoft-cs___cifar10/plain_text/0.0.0/0b2714987fa478483af9968de7c934580d0bb9a2/cifar10-train.arrow")],
        "img".to_string(),
        "label".to_string(),
    ).unwrap();
    println!("dataset length: {}", dataset.len());
    let img = dataset.get(0).unwrap();
    println!("first image label: {}", img.label);
    println!("first image encoded size: {} bytes", img.image.len());
    let image = image::load_from_memory(&img.image).unwrap();
    println!(
        "first image decoded size: {}x{}",
        image.width(),
        image.height()
    );
    let pipeline = rivet_core::pipeline::ImagePipeline::new(std::sync::Arc::new(dataset))
        .decode_image()
        .resize(8, 8)
        .unwrap()
        .normalize(vec![0.5; 3], vec![0.5; 3])
        .unwrap()
        .hwc_to_chw()
        .batch(64, false)
        .unwrap();
    let mut loader = pipeline.compile().unwrap();
    let batch = loader.next_batch().unwrap().unwrap();
    println!("batch size: {:?}", batch.shape);
}
