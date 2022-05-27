# Structure-Inverse-Design

1. Install needed packages using requirements.txt. Text the following command in terminal. You should run the code on Windows OS.
    ```
    pip3 install -r requirements.txt
    ```
    You have to install torch (**GPU version**) by yourself due to the difference in CUDA version.


2. For visualizing the structure design, you can download the software **PISA3D** (Platform of Inelastic Structural Analysis for 3D systems) in the following link:
https://www.ncree.org/PISA3D.aspx


3. The process of creating agents and environments are listed in *example.py*. You can also run *monteCarloDesign.py* to try the effect of improving the design.


4. You can find the input structure file (structure.ipt) and the given ground motions in *Files/Input*. Once you output the structure design or visualize the response, the result will be save in *Files/Output/{AI Algorithm}/{current date}* folder.


5. For the instructions of using ***Agent*** *(StructureDesigner)* and ***Environment*** *(StructureSimulator)*, they are listed in *FinalProject/Agent.pdf*.




