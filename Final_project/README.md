The script 'EQ2341_activity_recognition.ipynb' runs the whole project.
First, data for training and testing was imported and pre-processed. This is done by objects of the class 'DataHandler'.
The training data contains a sequence during which the three activities were performed several times in different orders. 
Furthermore, three files which capture one state each. They are used to estimate mean and covariance of the output distributions (B) of the HMM for the initalization.
After that, the HMM was created and its parameters {{q, A}, B} trained with the mentioned sequence using the Baum-Welch algorithm. This is handled by the function train() of class HMM.
Since testing sequences can begin with any state, the q vector is set back to equal probabilities.
Finally, the HMM was tested with three sequences of each activity and one sequence alternating between the three. The underlying states of a sequence were found by the viterbi() function of class HMM.
