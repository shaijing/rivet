use super::V2Config;
use crate::sample::image::{DecodedSample, ImageAxisOrder, ImageSample};
use crate::transforms::representation::channels::permute;
use crate::transforms::*;
use rivet_core::{DType, Device, Tensor};
use rivet_data::random::RandomStream;

mod v2_tests {
    use super::*;
    fn run(config: V2Config, image: Tensor, axis: ImageAxisOrder, seed: u64) -> Tensor {
        let mut rng = RandomStream::from_seed(seed);
        config
            .apply(
                ImageSample::Decoded(DecodedSample { image, label: 17 }),
                axis,
                &mut rng,
            )
            .unwrap()
            .into_decoded()
            .unwrap()
            .image
    }
    #[test]
    fn channel_permutations_read_strided_u8_and_float_without_changing_layout() {
        let image = Tensor::from_vec((0..36u8).collect::<Vec<_>>(), [3, 4, 3], &Device::Cpu)
            .unwrap()
            .narrow(1, 1, 2)
            .unwrap();
        for axis in [ImageAxisOrder::Hwc, ImageAxisOrder::Chw] {
            let input = if axis == ImageAxisOrder::Chw {
                image.permute(&[2, 0, 1]).unwrap()
            } else {
                image.clone()
            };
            let output = run(
                V2Config::PermuteChannels(vec![2, 0, 1]),
                input.clone(),
                axis,
                4,
            );
            let expected = if axis == ImageAxisOrder::Chw {
                vec![
                    5, 8, 17, 20, 29, 32, 3, 6, 15, 18, 27, 30, 4, 7, 16, 19, 28, 31,
                ]
            } else {
                vec![
                    5, 3, 4, 8, 6, 7, 17, 15, 16, 20, 18, 19, 29, 27, 28, 32, 30, 31,
                ]
            };
            assert_eq!(
                output.flatten_all().unwrap().to_vec1::<u8>().unwrap(),
                expected
            );
            let float = run(
                V2Config::PermuteChannels(vec![2, 0, 1]),
                input.to_dtype(DType::F32).unwrap(),
                axis,
                4,
            );
            assert_eq!(
                float.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                expected.into_iter().map(f32::from).collect::<Vec<_>>()
            );
        }
    }
    #[test]
    fn photometric_fixed_ranges_match_existing_color_sequence_and_sampling_order() {
        let image =
            Tensor::from_vec(vec![15u8, 40, 80, 100, 150, 210], [1, 2, 3], &Device::Cpu).unwrap();
        let config = V2Config::RandomPhotometricDistort {
            brightness: (0.9, 0.9),
            contrast: (1.1, 1.1),
            saturation: (0.8, 0.8),
            hue: (0.02, 0.02),
            probability: 1.0,
        };
        let seed = 8;
        let actual = run(config.clone(), image.clone(), ImageAxisOrder::Hwc, seed);
        let mut rng = RandomStream::from_seed(seed);
        for _ in 0..4 {
            assert!(rng.gen_bool(1.0).unwrap());
            rng.next_f64();
        }
        let before = rng.gen_bool(0.5).unwrap();
        assert!(rng.gen_bool(1.0).unwrap());
        let mut order = vec![0, 1, 2];
        for i in (1..3).rev() {
            let j = rng.gen_range_usize(0..i + 1).unwrap();
            order.swap(i, j);
        }
        let mut expected = ImageSample::Decoded(DecodedSample { image, label: 17 });
        expected = BrightnessConfig::new(0.9).apply(expected).unwrap();
        if before {
            expected = ContrastConfig::new(1.1).apply(expected).unwrap();
        }
        expected = SaturationConfig::new(0.8).apply(expected).unwrap();
        expected = HueConfig::new(0.02).apply(expected).unwrap();
        if !before {
            expected = ContrastConfig::new(1.1).apply(expected).unwrap();
        }
        let image = expected.into_decoded().unwrap().image;
        let expected = permute(&image, image.dims(), ImageAxisOrder::Hwc, &order).unwrap();
        assert_eq!(
            actual.flatten_all().unwrap().to_vec1::<u8>().unwrap(),
            expected.flatten_all().unwrap().to_vec1::<u8>().unwrap()
        );
    }
    #[test]
    fn zoom_out_uses_floor_offset_and_per_channel_fill() {
        let image = Tensor::from_vec(vec![5u8, 6, 7, 8, 9, 10], [1, 2, 3], &Device::Cpu).unwrap();
        let seed = 7;
        let mut rng = RandomStream::from_seed(seed);
        assert!(rng.gen_bool(1.0).unwrap());
        rng.next_f64();
        let left = (2.0 * rng.next_f64()).floor() as u32;
        let top = (1.0 * rng.next_f64()).floor() as u32;
        let expected = PadConfig::with_sides(left, top, 2 - left, 1 - top)
            .with_fill_channels([20.0, 30.0, 40.0])
            .apply(
                ImageSample::Decoded(DecodedSample {
                    image: image.clone(),
                    label: 17,
                }),
                ImageAxisOrder::Hwc,
            )
            .unwrap()
            .into_decoded()
            .unwrap()
            .image;
        let actual = run(
            V2Config::RandomZoomOut {
                fill: vec![20.0, 30.0, 40.0],
                side_range: (2.0, 2.0),
                probability: 1.0,
            },
            image,
            ImageAxisOrder::Hwc,
            seed,
        );
        assert_eq!(actual.dims(), [2, 4, 3]);
        assert_eq!(
            actual.flatten_all().unwrap().to_vec1::<u8>().unwrap(),
            expected.flatten_all().unwrap().to_vec1::<u8>().unwrap()
        );
    }
    #[test]
    fn probability_zero_preserves_storage_and_invalid_parameters_are_rejected() {
        let image = Tensor::from_vec(vec![1u8, 2, 3], [1, 1, 3], &Device::Cpu).unwrap();
        for config in [
            V2Config::RandomChannelPermutation { probability: 0.0 },
            V2Config::RandomZoomOut {
                fill: vec![0.0],
                side_range: (1.0, 4.0),
                probability: 0.0,
            },
            V2Config::RandomPhotometricDistort {
                brightness: (0.8, 1.2),
                contrast: (0.5, 1.5),
                saturation: (0.5, 1.5),
                hue: (-0.1, 0.1),
                probability: 0.0,
            },
        ] {
            let actual = run(config, image.clone(), ImageAxisOrder::Hwc, 1);
            assert!(actual.same_storage(&image));
        }
        for config in [
            V2Config::RandomResize {
                min_size: 2,
                max_size: 2,
                interpolation: InterpolationMode::Bilinear,
            },
            V2Config::RandomPhotometricDistort {
                brightness: (f32::NAN, 1.0),
                contrast: (0.5, 1.5),
                saturation: (0.5, 1.5),
                hue: (-0.1, 0.1),
                probability: 0.5,
            },
            V2Config::RandomZoomOut {
                fill: vec![1.0, 2.0],
                side_range: (1.0, 2.0),
                probability: 0.5,
            },
        ] {
            assert!(config.validate().is_err());
        }
    }
}

mod batch_contract_tests {
    use super::*;
    #[test]
    fn unscaled_batch_cast_requires_rank_four() {
        let input = Tensor::from_vec(vec![1u8; 3], [1, 1, 3], &Device::Cpu).unwrap();
        let config = V2Config::ToDtype {
            dtype: DType::F32,
            scale: false,
        };
        assert!(config.apply_batch(input, ImageAxisOrder::Hwc).is_err());
    }
}
