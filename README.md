# Adaptive-Systems-2021

## Assignments
Model-based Prediction and Control
```bash
Adaptive-Systems-2021/modelbased_prediction/
```


## Model-based Prediction and Control
### Installation
To install the package run the following command: 
```bash 
pip install .
```

If you are a developer you need to run this command:
```bash 
pip install .[dev]
```

## Run value based policy with the following command
Run the following command from `Adaptive-Systems-2021/` and press `any key` to go to the next state
```bash
python modelbased_prediction/code_for_implementation/simulator.py 1
```
#### Visual of the value based policy

![Alt Text](https://github.com/RichardDev01/Adaptive-Systems-2021/blob/main/assets/valuebasedpolicy.gif?raw=true)



### Iterations
<details open>
<summary> Click me </summary>


|it 0|  |   |    | 
|---|---|---|---|
|-1.|-1.|40.| 0.|
|-1.|-1.|-1.|40.|
|10.|-1.|-1.|-1.|
| 0.|10.|-1.|-1.|
|---|---|---|---|
|'u'|'r'|'r'|''|
|'d'|'u'|'u'|'u'|
|'d'|'l'|'d'|'u'|
|''|'l'|'l'|'u'|


|it 1|  |   |    | 
|---|---|---|---|
|-2.|39.|40.| 0.|
| 9.|-2.|39.|40.|
|10.| 9.|-2.|30.|
| 0.|10.| 8.|-2.|
|---|---|---|---|
|'r'|'r'|'r'|''|
|'d'|'u'|'u'|'u'|
|'d'|'l'|'u'|'u'|
|''|'l'|'l'|'u'|


|it 2|  |   |    | 
|---|---|---|---|
|38.|39.|40.| 0.|
| 9.|38.|39.|40.|
|10.| 9.|29.|30.|
| 0.|10.| 8.|29.|
|---|---|---|---|
|'r'|'r'|'r'|''|
|'u'|'u'|'u'|'u'|
|'d'|'u'|'u'|'u'|
|''|'l'|'u'|'u'|


|it 3|  |   |    | 
|---|---|---|---|
|38.|39.|40.| 0.|
|37.|38.|39.|40.|
|10.|37.|29.|30.|
| 0.|10.|28.|29.|
|---|---|---|---|
|'r'|'r'|'r'|''|
|'u'|'u'|'u'|'u'|
|'u'|'u'|'l'|'u'|
|''|'u'|'u'|'u'|


|it 4|  |   |    | 
|---|---|---|---|
|38.|39.|40.| 0.|
|37.|38.|39.|40.|
|36.|37.|36.|30.|
| 0.|36.|28.|29.|
|---|---|---|---|
|'r'|'r'|'r'|''|
|'u'|'u'|'u'|'u'|
|'u'|'u'|'l'|'l'|
|''|'u'|'u'|'u'|


|it 5|  |   |    | 
|---|---|---|---|
|38.|39.|40.| 0.|
|37.|38.|39.|40.|
|36.|37.|36.|35.|
| 0.|36.|35.|29.|
|---|---|---|---|
|'r'|'r'|'r'|''|
|'u'|'u'|'u'|'u'|
|'u'|'u'|'l'|'l'|
|''|'u'|'u'|'u'|


|it 6|  |   |    | 
|---|---|---|---|
|38.|39.|40.| 0.|
|37.|38.|39.|40.|
|36.|37.|36.|35.|
| 0.|36.|35.|34.|
|---|---|---|---|
|'r'|'r'|'r'|''|
|'u'|'u'|'u'|'u'|
|'u'|'u'|'l'|'l'|
|''|'u'|'u'|'u'|


</details>

### Run random based policy with the following command
Run the following command from `Adaptive-Systems-2021/` and press `any key` to go to the next state
```bash
python modelbased_prediction/code_for_implementation/simulator.py 0
```
#### Visual of the random based policy
![Alt Text](https://github.com/RichardDev01/Adaptive-Systems-2021/blob/main/assets/randompolicy.gif?raw=true)