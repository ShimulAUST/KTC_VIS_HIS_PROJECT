# How to Add a New Algorithm to KTC-Vis

This guide walks through adding a 4th reconstruction algorithm to the KTC-Vis dashboard.

## Step 1: Create an Adapter

Create a new file in `ktc_vis/adapters/`:

```python
# ktc_vis/adapters/my_algo_adapter.py
from ktc_vis.adapters.base import AlgorithmAdapter
from ktc_vis.data.loader import KTCMeasurement
import numpy as np

class MyAlgoAdapter(AlgorithmAdapter):
    name = "my_algo"

    def reconstruct(self, measurement: KTCMeasurement) -> np.ndarray:
        # Your implementation here
        # MUST return a 256×256 uint8 array with values in {0, 1, 2}
        result = ...  # run your algorithm
        return result.astype(np.uint8)
```

## Step 2: Register in Config

Add your algorithm to `configs/experiment.yaml`:

```yaml
benchmark:
  algorithms:
    - abc1
    - cuqi8
    - pnpe2e
    - my_algo   # ← add here
```

Also add any weight paths if needed:

```yaml
data:
  weights:
    my_algo: "data/raw/weights/my_algo"
```

## Step 3: Load in run_benchmark.py

Add an `elif` branch in `scripts/run_benchmark.py → load_adapter()`:

```python
elif algorithm == "my_algo":
    from ktc_vis.adapters.my_algo_adapter import MyAlgoAdapter
    return MyAlgoAdapter(weights_dir=config["data"]["weights"]["my_algo"])
```

## Step 4: Run the Benchmark

```bash
python scripts/run_benchmark.py \
    --config configs/experiment.yaml \
    --algorithms my_algo
```

This will populate the HDF5 cache for your new algorithm across all levels and samples.

## Step 5: The Dashboard Updates Automatically

The sidebar dropdown and all 6 modules read algorithm names from the cache and config — no further UI changes required.

## Step 6: Write Tests

Add integration tests in `tests/test_adapters.py`:

```python
@pytest.mark.requires_data
def test_my_algo_output_shape(dummy_measurement):
    adapter = MyAlgoAdapter(weights_dir="data/raw/weights/my_algo")
    result = adapter.reconstruct(dummy_measurement)
    adapter.validate_output(result)  # Checks shape, dtype, values
```
