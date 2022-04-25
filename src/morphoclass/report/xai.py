"""Implementation of the ``XAIReport`` class."""
from __future__ import annotations

import logging
import os
import pathlib
import textwrap

import matplotlib

from morphoclass.report import plumbing

logger = logging.getLogger(__name__)


class XAIReport:
    """XAI report manager."""

    def __init__(self, name: str, base_dir: str | os.PathLike) -> None:
        if name.lower().endswith(".html"):
            raise ValueError(
                "The report name must not contain the file extension and "
                f'therefore cannot end on ".html". Got name: {name!r}'
            )
        self.base_dir = pathlib.Path(base_dir).resolve()
        self.img_dir = self.base_dir / f"{name}-images"
        self.report_path = (self.base_dir / name).with_suffix(".html")
        self.sections = {}
        self.titles = {}
        self.figures = []  # (fig, file_stem)

    def add_section(self, title: str, _id: str, section_html: str) -> None:
        """Add a new XAI section to the report."""
        _id = _id.replace(" ", "-")
        self.titles[_id] = title.strip()
        self.sections[_id] = textwrap.dedent(section_html).strip()

    def add_figure(self, fig: matplotlib.figure.Figure, name: str) -> pathlib.Path:
        """Add a new figure to the report.

        The figure will be kept in memory until the ``write()`` call. It
        is only when ``write()`` is called that the figures will be written
        to disk together with the actual report.

        This method returns a path that can be used to reference the
        figure image file on disk.

        Parameters
        ----------
        fig
            A matplotlib figure.
        name
            The name of the figure. This will be used as the file name of
            the image file saved to disk. The name should not contain
            a file extension - it will be added automatically.

        Returns
        -------
        pathlib.Path
            The path under which the image will be stored on disk. This path
            is relative to the base directory for reproducibility reasons
            and can be used to refer to the figure image file in HTML blocks.
        """
        fig_path = (self.img_dir / name).with_suffix(".png")
        self.figures.append((fig, fig_path))

        return fig_path.relative_to(self.base_dir)

    @property
    def _template_vars(self):
        toc_lines = []
        section_blocks = []
        for _id, toc_title in self.titles.items():
            toc_lines.append(f"<a href='#{_id}'>{toc_title}</a>")

            # Prepend a section header to each section block
            section_header = f"""
            <div class="page-header">
                <h1 id="{_id}">{self.titles[_id]}</h1>
            </div>
            """
            section_header = textwrap.dedent(section_header).strip()
            section_blocks.append(f"{section_header}<br/>{self.sections[_id]}")
        toc_html = "<br/>".join(toc_lines)
        report_html = "<br/><br/><hr/><br/>".join(section_blocks)

        return {"toc_html": toc_html, "report_html": report_html}

    def write(self):
        """Render and write the XAI report to disk."""
        template = plumbing.load_template("xai-report")
        self.base_dir.mkdir(exist_ok=True, parents=True)

        logger.info(f"Writing {len(self.figures)} figures")
        self.img_dir.mkdir(exist_ok=True)
        for fig, file_name in self.figures:
            fig.savefig(file_name)

        logger.info(f"Writing report to {self.report_path.as_uri()}")
        plumbing.render(template, self._template_vars, self.report_path)
