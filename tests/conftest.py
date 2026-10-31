# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

## Small fits run ~3x faster on one thread; results are identical.
os.environ.setdefault("OMP_NUM_THREADS", "1")
