# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from importlib.metadata import version

from pappi.core import *  # noqa: F403
from pappi.core import __all__  # noqa: F401

__version__ = version("pappi")
