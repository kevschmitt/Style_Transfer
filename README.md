# How to run

## Python environment for flask backend
First make a python virtual environment

To do this, navigate to the project folder and use the command: python -m venv myenv

then we need to activate the environment using the command: source bin/activate

Now the python environment is set up we need to download the dependencies use the command: pip install -r requirements.txt

Then we need to activate the flask backed using the command: flask run

In the command line it will give you a URL endpoint to access the flask API

coppy this  and make sure that it matches the url on line 21 of the src/App.js file if it does not match, paste it in and save.

# Open Front end

First make sure you have npm installed.

If you do not have NPM installed in a separate terminal can use: brew install node

Then to give access to your node scripts you need to execute the following commands

ls -l node_modules/.bin/react-scripts

and 

chmod +x node_modules/.bin/react-scripts


Then to start the front end use: NPM start

Then the webpage should open in your default web browser, or you can go to http://localhost:3000/ in a web browser to 
get to the website. 

# Use Application
Input two images (they should be the same exact size for best results), click combine and wait ~15-25 Mins for the combined image to pop up.