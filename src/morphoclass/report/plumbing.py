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
"""Module for creating Jinja reports."""
from __future__ import annotations

import logging
import pathlib

import jinja2.meta
from jinja2 import Environment
from jinja2 import PackageLoader
from jinja2 import Template
from jinja2 import TemplateNotFound

logger = logging.getLogger()


def get_loader() -> jinja2.PackageLoader:
    """Get a jinja loader for morphoclass report templates.

    Returns
    -------
    The jinja package loader for morphoclass.
    """
    return PackageLoader("morphoclass")


def list_templates() -> set[str]:
    """List the available jinja templates in morphoclass.

    Returns
    -------
    The set of template names in morphoclass.
    """
    file_names = get_loader().list_templates()
    template_names = {name[:-5] for name in file_names if name.endswith(".html")}
    return template_names


def template_variables(template_name: str) -> set[str] | None:
    """List the template variables for a given template.

    Parameters
    ----------
    template_name
        The name of the template (see `list_templates()`).

    Returns
    -------
    If the template exists then a set of variable names that can be
    interpolated into the template are returned. Otherwise, None is
    returned.
    """
    template_file_name = template_name + ".html"
    env = Environment()
    loader = get_loader()
    try:
        template_source, *_ = loader.get_source(env, template_file_name)
    except TemplateNotFound:
        return None
    ast = env.parse(template_source)
    variables = jinja2.meta.find_undeclared_variables(ast)

    return set(variables)


def load_template(template_name: str) -> Template | None:
    """Load a given jinja template.

    Parameters
    ----------
    template_name
        The name of the template (see `list_templates()`).

    Returns
    -------
    A jinja template.
    """
    template_file_name = template_name + ".html"
    loader = get_loader()
    env = Environment(loader=loader)
    try:
        template = env.get_template(template_file_name)
    except TemplateNotFound:
        return None

    return template


def render(template: Template, template_vars: dict, output_path: pathlib.Path) -> None:
    """Generate a jinja HTML report.

    Parameters
    ----------
    template
        The template to render.
    template_vars
        The template variables to interpolate into the template.
    output_path
        The path under which to save the HTML report.
    """
    html = template.render(template_vars)
    with output_path.open("w") as fh:
        fh.write(html)
