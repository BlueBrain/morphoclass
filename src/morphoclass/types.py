# Copyright © 2022-2022 Blue Brain Project/EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Custom type definitions."""
from __future__ import annotations

import os
from typing import Union

# Starting py39 should write `os.PathLike[str]`
# See also
# https://github.com/samuelcolvin/pydantic/blob/9d631a3429a66f30742c1a52c94ac18ec6ba848d/pydantic/typing.py#L165
# https://github.com/python/typing/issues/402
# if type(x) = StrPath, then one should be able to write x = pathlib.Path(x)
StrPath = Union[str, os.PathLike]
