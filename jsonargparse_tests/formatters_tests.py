#!/usr/bin/env python3

from io import StringIO
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

        out = StringIO()
        parser.print_help(out)
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

        self.assertIn('--g2.v4 V4', outval)
        self.assertIn('APP_G2__V4', outval)

        self.assertIn('--v5, --no_v5', outval)
        self.assertIn('APP_V5', outval)
        self.assertIn('Option v5. (type: bool, default: True)', outval)


class DefaultFormatterTmpdirTests(TempDirTestCase):

    def test_default_config_files_help(self):
        not_exist = 'does_not_exist.yaml'
        config_path = os.path.realpath(os.path.join(self.tmpdir, 'config.yaml'))
        with open(config_path, 'w') as output_file:
            output_file.write('v1: from yaml v1\nn1.v2: from yaml v2\n')

        parser = ArgumentParser(default_config_files=[not_exist, config_path])
        parser.add_argument('--v1', default='from default v1')
        parser.add_argument('--n1.v2', default='from default v2')

        out = StringIO()
        parser.print_help(out)
        outval = out.getvalue()

        self.assertIn('default config file locations', outval)
        self.assertIn('from yaml v1', outval)
        self.assertIn('from yaml v2', outval)
        self.assertIn(str([not_exist, config_path]), outval)
        self.assertIn(config_path, outval)

        parser.default_config_files = [not_exist]
        out = StringIO()
        parser.print_help(out)
        outval = out.getvalue()

        self.assertIn('from default v1', outval)
        self.assertIn('from default v2', outval)
        self.assertIn(str([not_exist]), outval)
        self.assertIn('no existing default config file found', outval)

        parser.default_config_files = None
        out = StringIO()
        parser.print_help(out)
        outval = out.getvalue()

        self.assertNotIn('default config file locations', outval)


    def test_default_config_files_help_with_required(self):
        config_path = os.path.realpath(os.path.join(self.tmpdir, 'config.yaml'))
        with open(config_path, 'w') as output_file:
            output_file.write('v1: from yaml\n')

        parser = ArgumentParser(default_config_files=[config_path])
        parser.add_argument('req', help='req description')
        parser.add_argument('--v1', default='from default')

        out = StringIO()
        parser.print_help(out)
        outval = out.getvalue()

        self.assertIn('req description', outval)
        self.assertIn('from yaml', outval)


if __name__ == '__main__':
    unittest.main(verbosity=2)
