### Codes for the paper titled "A slack-based two-stage improved particle swarm optimization algorithm for robust scheduling of a flexible job-shop with new and remanufacturing jobs" (ESWA-D-24-12709.R1, under review after revision).
### Install: 
Please install the required dependencies using "pip install requirements.txt".
### Usage:
1. The "Benckmark Instance" folder contains a total of 25 instances from the three types of benchmark datasets (Brandimarte/Fattahi/Kacem) used in this paper.
2. Run "STIPSO.py" file to get robust solution with optimal makespan (Under default settings in our paper). You can also change the example to suit your needs.
3. Run "TS_for_Benchmark.py" file to calculate the TS value of the solution obtained by the ST-IPSO algorithm. Likewise, you can change the examples or adjust the parameters to suit your needs.
4. Run "MonteCarlo_for_Benchmark.py" file to calculate the RM value of the solution obtained by the ST-IPSO algorithm. Likewise, you can change the examples or adjust the parameters to suit your needs.
5. The "Get_Problem.py" file provides a function to convert the publicly available datasets (Brandimarte/Fattahi/Kacem) from the.txt format to the matrix format needed as input to the ST-IPSO algorithm.