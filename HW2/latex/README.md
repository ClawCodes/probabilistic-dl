# Report files

Compile `main.tex` from this directory. Every generated figure is saved by the
Python pipeline directly into `figs/`, which is the only image path LaTex uses.

Suggested runs:

```powershell
python -m src.run --dataset two_moons --method ensemble --epochs 200 --members 5
python -m src.sweep --dataset two_moons --sizes 1 2 5 10 --epochs 200
python -m src.run --dataset fashion_mnist --method ensemble --epochs 10 --members 5
```
