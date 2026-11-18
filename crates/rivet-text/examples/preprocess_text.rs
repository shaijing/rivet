//! An offline text-classification pipeline; optionally pass a tokenizer.json path.
use rivet_text::{PaddingParams, TextPipeline, TextSample, TextTokenizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = match std::env::args_os().nth(1) {
        Some(path) => TextTokenizer::from_file(path)?,
        None => TextTokenizer::from_json(include_str!("../tests/fixtures/wordlevel.json"))?,
    };
    let pad_id = tokenizer
        .token_to_id("[PAD]")
        .ok_or("example requires a [PAD] token; configure the model's padding token explicitly")?;
    let pipeline = TextPipeline::from_samples(
        vec![
            TextSample::new("Hello world").with_label(1),
            TextSample::new("Good").with_label(0),
            TextSample::new("Bad world").with_label(0),
        ],
        tokenizer,
    )
    .truncate(32)
    .padding(PaddingParams {
        pad_id,
        ..Default::default()
    })
    .batch(2, false);
    println!("{}", pipeline.explain()?);
    for batch in pipeline.compile()? {
        let batch = batch?;
        println!(
            "source_indices={:?} shape={:?}",
            batch.source_indices,
            batch.input_ids.dims()
        );
        println!("input_ids={:?}", batch.input_ids.to_vec::<i64>()?);
        println!("attention_mask={:?}", batch.attention_mask.to_vec::<i64>()?);
        if let Some(labels) = batch.labels {
            println!("labels={:?}", labels.to_vec::<i64>()?);
        }
    }
    Ok(())
}
