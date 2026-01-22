## Dataset

This project uses the **Labeled Faces in the Wild (LFW)** dataset for experimental
evaluation.

Instead of relying on an external download link, the dataset is loaded directly
using `scikit-learn`, which automatically handles downloading and caching:

```python
from sklearn.datasets import fetch_lfw_people
```
