# Speed Profile

This project computes a motorcycle speed profile around a race track.

The core script, `speed_profile.py`, loads a CSV file containing `x,y`
coordinates that describe a racing line. The path is resampled to a fixed
spacing and a simple physics model estimates the maximum speed, gear and engine
RPM at each point. The result is written to a new CSV file along with the
cumulative distance travelled.

## Usage

```
python speed_profile.py input.csv output.csv [options]
```

### Common options

* `--step` – resampling distance in metres (default: 2)
* `--traction-circle` – limit acceleration by lateral grip
* `--trail-braking` – apply traction limit while braking
* `--sweeps` – number of forward/backward passes for the solver
* Motorcycle parameters such as `--m` (mass) and `--gears` can be overridden; see
  `python speed_profile.py --help` for the full list.

The output CSV contains `x`/`y` coordinates, cumulative distance, speed in both
m/s and km/h, selected gear and engine RPM.

## Development notes

The physics model and solver are contained in `compute_speed_profile` and the
`BikeParams` dataclass. To experiment with different motorcycles or track
conditions, adjust the fields in `BikeParams` or expose additional command-line
arguments in `main`.

Tests are not provided; when modifying the code, run

```
python speed_profile.py --help
```

to ensure the script loads correctly.