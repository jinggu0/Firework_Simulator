from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.opengl
def test_default_framebuffer_capture_preserves_rgb_and_has_no_gl_error():
    # Separate SDL context so this also checks the real default framebuffer,
    # which standalone/offscreen contexts do not expose.
    code = """
import moderngl
import numpy as np
import pygame
from simulator.validation.capture import read_display_sdr
pygame.init()
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
pygame.display.set_mode((32, 24), pygame.OPENGL | pygame.DOUBLEBUF | pygame.HIDDEN)
ctx = moderngl.create_context(require=330)
try:
    ctx.screen.use()
    ctx.screen.clear(.25, .5, .75, 1)
    assert ctx.error == 'GL_NO_ERROR'
    ctx.scissor = (1, 1, 2, 2)
    for _ in range(2):
        frame = read_display_sdr(ctx)
        assert frame.shape == (24, 32, 3)
        assert np.max(np.abs(frame.astype(int) - [64, 128, 191])) <= 1
        assert ctx.error == 'GL_NO_ERROR'
        assert ctx.fbo is ctx.screen
        assert ctx.scissor == (1, 1, 2, 2)
finally:
    ctx.release()
    pygame.quit()
"""
    process = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
