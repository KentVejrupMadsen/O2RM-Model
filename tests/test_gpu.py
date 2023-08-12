import tensorflow


# Will fail if not present (and packages not available)
def test_has_gpu_present() -> bool:
    return_value: bool = False
    physical_devices = tensorflow.config.list_physical_devices('GPU')

    for p in physical_devices:
        print(
            p.name,
            '\n'
        )

    assert (len(physical_devices) > 0)
    return return_value
