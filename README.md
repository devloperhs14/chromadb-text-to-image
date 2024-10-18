# Text To Image
A  slef hosted simple app that utilises chromadb's vector search to find the text image similarity and return using gradio interface. Very easy to use and intutive. Follow installation instructions to use for your own.

# Get Started
Run the following command in sequence in linux shell or mac terminal!

## 1. Clone repo & cd

`git clone https://github.com/devloperhs14/text-to-image-app.git`

`cd repo_name`

Open the repo in any of the IDE. then!


## 2. Create a environment and activate it
`python3 -m venv .venv`

`source .venv/bin/activate` - linux or mac

`.venv\Scripts\activate` - windows

## 3. Install Requirements.txt
`pip install -r requirements.txt`

# Usage
`python3 text-to-image.py`

If you want to include your own image to test, follow the steps:
* Put your image in img folder
* Copy and paste the path of the image in `text-to-img.py` file under **image path**
* Rerun the **text-to-image.py** file again

# Trobuleshoot Issues
**Windows** 
One of the library is not supported, so windows is not recomended for the project

**SQL Lite 3 error from chromadb!**
For fixing this follow the following steps (on linux or mac)
* Install the sql lite 3 latest binary using `pip install pysqlite3-binary`
* Navigate to `venv3.10/lib/python3.10/site-packages/chromadb/__init__.py`
* Paste the following code at the begining of the `__init__.py` file and **save**:
```
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
```

# Future Direction
Adding support to handle self loading images from directory, so user dont have to provide the path!

* Rerun : `python3 text-to-image.py`

