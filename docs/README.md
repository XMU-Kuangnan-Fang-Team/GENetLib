GENetLib's Website Development Guide
---------

## Prerequisite
Clone the source code of ``GENetLib`` into your device:
```bash
git clone git@github.com:Barry57/GENetLib.git
cd GENetLib
```
Install dependent libraries:
```bash
cd docs
pip install -r requirements.txt
```

## Generate Website's Content  
Conduct the following command to generate the content of website:
```bash
make html
```
You can check the website's contents by opening `index.html` file
in the directory `docs/build/html` with your browser.
For more components and details, please refer to [documatt](https://documatt.com/sphinx-themes/).