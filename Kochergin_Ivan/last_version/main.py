import moduls.model
import sys
import os
from streamlit import __main__ as stcli
#os.system('fuser -k 8500/tcp')
model = moduls.model.edit_model()
if __name__ == '__main__':
    sys.argv = ["streamlit", "run", '.\Corn\streamlit_app.py', '--theme.base', 'dark', '--theme.primaryColor', '#F59A07', '--theme.font', 'serif']
    sys.exit(stcli.main())

