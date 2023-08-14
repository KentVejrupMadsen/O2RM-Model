class MomentumDecay:
    def __init__(
        self, 
        start_learning_value: float = 0.004,
        steps: int = 2
    ):
        self.value: float = start_learning_value
        self.steps: int = steps

        self.minimum: float | None = None
        self.maximum: float | None = None

    def __del__(
        self
    ):
        del                     \
            self.value,         \
            self.steps,         \
            self.minimum,       \
            self.maximum

    def set_value(
        self, 
        with_value: float
    ) -> None:
        self.value = with_value

    def get_value(
        self
    ) -> float:
        return self.value
    
    def get_steps(self) -> int:
        return self.steps
    
    def set_steps(
        self, 
        value:int
    ) -> None:
        self.steps = value

    def __float__(
        self
    ) -> float:
        return self.get_value()
    
    def update(
            self,
            input_list: list
    ) -> None | float:
        size_of_list: int = len(
            input_list
        )

        last_index: int = size_of_list - 1
        calculated_momentum: float = 0.0
        counted_steps: int = 0

        latest_data: list = reversed(
            input_list
        )

        for i in range(
            self.get_steps()
        ):
            if i > last_index:
                break
            value = latest_data[i]

            if self.maximum is None or value > self.maximum:
                self.maximum = value

            if self.minimum is None or value < self.minimum:
                self.minimum = value

            calculated_momentum = calculated_momentum + value
            counted_steps = counted_steps + 1
        
        calculated_momentum = float(
            float(
                calculated_momentum
            ) 
            / 
            float(
                counted_steps
            )
        )

        value_range: float = self.maximum - self.minimum
        multiplier: float = 1 * value_range

        if multiplier > 1.0:
            multiplier = 1 / multiplier

        self.set_value(
            self.get_value() - (calculated_momentum * multiplier)
        )

        return calculated_momentum
    