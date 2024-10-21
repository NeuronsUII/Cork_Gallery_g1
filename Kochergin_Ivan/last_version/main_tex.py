import sys
from streamlit import __main__ as stcli



if __name__ == '__main__':
    sys.argv = ["streamlit", "run", '.\Corn\streamlit_for_texture.py', '--theme.base', 'dark', '--theme.primaryColor', '#F59A07', '--theme.font', 'serif']
    sys.exit(stcli.main())