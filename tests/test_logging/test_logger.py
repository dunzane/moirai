# pylint: disable=invalid-name, inconsistent-quotes, unused-argument,
# pylint: disable=function-redefined, use-implicit-booleaness-not-comparison
import unittest
import logging as std_logging
import tempfile
import os
from unittest.mock import patch

from pipeai.logging import get_logger, logger_initialized


class TestGetLogger(unittest.TestCase):
    """Unit tests for the get_logger function"""

    def setUp(self) -> None:
        for name in list(logger_initialized):
            logger = get_logger(name)
            logger.handlers.clear()
            logger_initialized.remove(name)

    @patch('pipeai.logging.logger.is_master', return_value=False)
    def test_stream_handler_added(self, mock_is_master):
        logger = get_logger("test_logger")

        assert any(isinstance(h, std_logging.StreamHandler) for h in logger.handlers)

    @patch('pipeai.logging.logger.is_master', return_value=True)
    def test_file_handler_added_when_master(self, mock_is_master):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        logger = get_logger("file_logger", log_file=tmp_path)
        assert any(isinstance(h, std_logging.FileHandler) for h in logger.handlers)
        os.remove(tmp_path)

    @patch('pipeai.logging.logger.is_master', return_value=False)
    def test_file_handler_not_added_when_not_master(self, mock_is_master):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        logger = get_logger("no_file_logger", log_file=tmp_path)
        assert not any(isinstance(h, std_logging.FileHandler) for h in logger.handlers)
        os.remove(tmp_path)

    @patch('pipeai.logging.logger.is_master', return_value=True)
    def test_logger_level_master_vs_non_master(self, mock_is_master):
        logger = get_logger("master_logger", log_level=std_logging.DEBUG)
        assert logger.level == std_logging.DEBUG

        mock_is_master = False
        logger = get_logger("worker_logger", log_level=std_logging.DEBUG)
        assert not logger.level == std_logging.ERROR

    @patch('pipeai.logging.logger.is_master', return_value=True)
    def test_logger_reuse(self, mock_is_master):
        logger1 = get_logger("reuse_logger")
        logger2 = get_logger("reuse_logger")
        assert logger1 is logger2
        assert "reuse_logger" in logger_initialized
