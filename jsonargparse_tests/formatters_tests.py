#!/usr/bin/env python3

from io import StringIO
from contextlib import redirect_stdout
from jsonargparse_tests.base import *


class DefaultFormatterTests(unittest.TestCase):

    def test_default_help_formatter(self):
        parser = ArgumentParser(prog='app', default_env=True)
        parser.add_argument('--cfg', help='Config in yaml/json.', action=ActionConfigFile)
        parser.add_argument('--v1', help='Option v1.', default='v1', required=True)
        parser.add_argument('--v2', type=int, help='Option v2.')
        parser.add_argument('--g1.v3', help='Option v3.', default='v3')
        parser.add_argument('--v4', choices=['A', 'B'], help='Option v4.')
        parser2 = ArgumentParser()
        parser2.add_argument('--v4')
        parser.add_argument('--g2', action=ActionParser(parser=parser2))
        parser.add_argument('--v5', action=ActionYesNo, default=True, help='Option v5.')

        os.environ['COLUMNS'] = '150'
        out = StringIO()
        with redirect_stdout(out):
            parser.print_help()
        outval = out.getvalue()

        self.assertIn('--print_config', outval)

        self.assertIn('--cfg CFG', outval)
        self.assertIn('APP_CFG', outval)
        self.assertIn('Config in yaml/json.', outval)

        self.assertIn('--v1 V1', outval)
        self.assertIn('APP_V1', outval)
        self.assertIn('Option v1. (required, default: v1)', outval)

        self.assertIn('--v2 V2', outval)
        self.assertIn('APP_V2', outval)
        self.assertIn('Option v2. (type: int, default: null)', outval)

        self.assertIn('--g1.v3 V3', outval)
        self.assertIn('APP_G1__V3', outval)
        self.assertIn('Option v3. (default: v3)', outval)

        self.assertIn('--v4 {A,B}', outval)
        self.assertIn('APP_V4', outval)
        self.assertIn('Option v4. (default: null)', outval)

        self.assertIn('--g2 G2', outval)
        self.assertIn('APP_G2', outval)
        self.assertIn('--g2.help', outval)

        self.assertIn('--v5, --no_v5', outval)
        self.assertIn('APP_V5', outval)
        self.assertIn('Option v5. (type: bool, default: True)', outval)


if __name__ == '__main__':
    unittest.main(verbosity=2)
