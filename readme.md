**Aufbau Repository**
- Belegarbeit "Beleg_Datenkompression.pdf" im root Verzeichnis
- Quellcode des entwickelten Tools in /src und /bin Verzeichnissen
- Testbilder sowie Analysebilder welche im Beleg verwendet sind befinden sich in den Verzeichnissen unter /latex/images/...

**Requirement:**
- Python >= 3.9

**Setup**
- use linux or install Ubuntu WSL from Windows 10 Store 
- update package repos and upgrade packages
   - `sudo apt update`
   - `sudo apt upgrade`
- clone the project using git
- install Python, Python virtual environment and basic packages:
  - `sudo apt install python3.9 python3.9-dev python3.9-venv build-essential`
- go into project folder and enter:
  - `python3.9 -m venv ./venv`
  - `. venv/bin/activate`
  - `pip install Cython==0.29.23 numpy==1.20.3 setuptools wheel`
  - `pip install .`
- done

**Usage**
- `jpeg-cli -h` to show available options
