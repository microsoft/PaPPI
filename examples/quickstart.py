# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from rich.console import Console

from pappi import Protocol, audit
from pappi.demo import synthetic_campaign, true_score

frame = synthetic_campaign()
report = audit(frame, protocols=[Protocol("roadside", ["dist_to_road"])], light=True,
               progress=True)
Console().print(report)
print(f"true deployment macro-F1: {true_score(frame, report.models):.2f} "
      "(from the hidden labels)")
