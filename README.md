## Local Dev
```shell script
python3 -m venv ~/.virtualenvs/tictactoe
```

## Docker
To get gui to run you need to run
```shell script
xhost +local:docker
```

For security reasons you should follow up with 
```shell script
xhost -local:docker
```