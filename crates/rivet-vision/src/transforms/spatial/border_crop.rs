pub(crate) fn sample_border_crop(
    height: usize,
    width: usize,
    [left, right, top, bottom]: [f64; 4],
    rng: &mut rivet_data::random::RandomStream,
) -> crate::errors::RivetResult<[u32; 4]> {
    if height == 0 || width == 0 || height > u32::MAX as usize || width > u32::MAX as usize {
        return Err(crate::errors::invalid_shape(
            "RandomCropFromBorders requires nonempty dimensions that fit u32",
        ));
    }
    let mut axis =
        |length: usize, start: f64, end: f64| -> crate::errors::RivetResult<(u32, u32)> {
            let start_max = ((length as f64 * start).floor() as usize).min(length - 1);
            let first = rng.gen_range_usize(0..start_max + 1)?;
            let end_max = ((length as f64 * end).floor() as usize).min(length - first - 1);
            let last = rng.gen_range_usize(0..end_max + 1)?;
            Ok((first as u32, (length - first - last) as u32))
        };
    let (x, w) = axis(width, left, right)?;
    let (y, h) = axis(height, top, bottom)?;
    Ok([x, y, w, h])
}
