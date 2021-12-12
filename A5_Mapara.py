'''
Name: Maha Mapara
CS 131-A5: Naive Bayesian Classification
Due date: 11/29/21
'''

#################################################libraries needed#########################################################################
import pandas as pd
import math as m

################################################RBE class###################################################################################
#Contains the meain Recursive Bayesian Estimation function and other helper function used by RBE
class RBE:
    def __init__(self, pdf, data, obj):
        '''
        Purpose: Creates an RBE object

        Arguments:
            pdf = a pandas dataframe containing the probability of being an airplane or bird based on speed
            data = a pandas dataframe containing the speed of 10 objects/tracks at 300 time points. (10 rows, 300 columns)
            obj = the object number (can be 0 to 9)
        '''
        self.pdf = pdf
        self.data = data
        self.obj = obj

    def get_probs(self, time):
        '''
        Purpose: get probabilities of being a plane or bird based from pdf.txt on speed value from data.txt file

        Arguments:
            pdf = a pandas dataframe containing the probability of being an airplane or bird based on speed
            data = a pandas dataframe containing the speed of 10 objects/tracks at 300 time points. (10 rows, 300 columns)
            obj = the object number (can be 0 to 9)
            time = the time point for which we want the speed

        Returns:
            prob_bird = probability of being a bird for that speed value from pdf
            prob_plane= probability of being a bird for that speed value from pdf
        '''
        speed = self.data.iloc[self.obj][time] #get speed using object number and time point
        speed_search = (m.ceil(speed))*2 #to get correct index in pdf df, round and double
        prob_bird = self.pdf[speed_search - 1][0] #-1 for 0 indexing. row 0 is bird
        prob_plane = self.pdf[speed_search - 1][1] #-1 for 0 indexing. row 1 is plane

        return prob_bird, prob_plane

    def get_bird_prob(self, time):
        '''
        Purpose: Get prob of being a bird

        Arguments:
            pdf = a pandas dataframe containing the probability of being an airplane or bird based on speed
            data = a pandas dataframe containing the speed of 10 objects/tracks at 300 time points. (10 rows, 300 columns)
            obj = the object number (can be 0 to 9)
            time = the time point for which we want the speed

        Returns:
            prob of being a bird at the time point
        '''
        return self.get_probs(time)[0] #get only bird prob

    def get_plane_prob(self, time):
        '''
        Purpose: Get prob of being a plane

        Arguments:
            pdf = a pandas dataframe containing the probability of being an airplane or bird based on speed
            data = a pandas dataframe containing the speed of 10 objects/tracks at 300 time points. (10 rows, 300 columns)
            obj = the object number (can be 0 to 9)
            time = the time point for which we want the speed

        Returns:
            prob of being a plane at the time point
        '''
        return self.get_probs(time)[1] #get only plane prob

    def rbe(self):
        '''
        Purpose: Runs Recusive Bayesian Estimation

        Arguments:
            pdf = a pandas dataframe containing the probability of being an airplane or bird based on speed
            data = a pandas dataframe containing the speed of 10 objects/tracks at 300 time points. (10 rows, 300 columns)
            obj = the object number (can be 0 to 9)

        Returns:
            probs_bird = The probability of an object/track being a bird for all 300 time points
            probs_plane = The probability of an object/track being a plane for all 300 time points

        '''
        #create empty lists which will store all 300 probabilties
        probs_bird = []
        probs_plane = []

        ########for t = 0
        prior = 0.5
        #calc belief of state at t = 0
        B0_b = self.get_bird_prob(0)*prior #belief for bird
        B0_p = self.get_plane_prob(0)*prior #belief for plane
        #normalize B0- divide beliefs by total sum of beliefs
        sum_B0 = B0_b + B0_p
        #get probs
        P0_b = B0_b/sum_B0 #prob of bird at t = 0
        P0_p = B0_p/sum_B0 #prob of plane at t = 0

        #append to list
        probs_bird.append(P0_b)
        probs_plane.append(P0_p)

        #########the remaining t
        for i in range(1,300): #from t = 1 to 299
            #calc belief when state is bird at that t
            prob_OS_b = self.get_bird_prob(i) #P(O|S) when state is bird
            sum_part_b = (0.9*probs_bird[i-1]) + (0.1*probs_plane[i-1]) #prob of being bird or plane at last time point times 0.9 or 0.1
            Bt_b = prob_OS_b*sum_part_b

            #calc belief when state is plane at that t
            prob_OS_p = self.get_plane_prob(i) #P(O|S) when state is plane
            sum_part_p = (0.9*probs_plane[i-1]) + (0.1*probs_bird[i-1]) #prob of being bird or plane at last time point times 0.9 or 0.1
            Bt_p = prob_OS_p*sum_part_p

            #normalize Bt- divide beliefs by total sum of beliefs
            sum_Bt = Bt_b + Bt_p
            #get probs
            Pt_b = Bt_b/sum_Bt
            Pt_p = Bt_p/sum_Bt

            #attend probs for each t
            probs_bird.append(Pt_b)
            probs_plane.append(Pt_p)

        return probs_bird, probs_plane

###################################################RBEWork##############################################################################
#Inherits RBE, puts the probabilties of being a bird or airplane at every time point in data frames
class RBEWork(RBE):
    def print_probs_nicely(self):
        '''
        Purpose: To put all bird and plane probabilties in a data frame

        Arguments:
            pdf = a pandas dataframe containing the probability of being an airplane or bird based on speed
            data = a pandas dataframe containing the speed of 10 objects/tracks at 300 time points. (10 rows, 300 columns)
            obj = the object number (can be 0 to 9)

        Returns:
            bird_plane = dataframe of 3 columns: P(Bird|Obs), P(Plane|Obs) and Time
        '''

        prob_list_b = self.rbe()[0] #prob list for bird
        prob_list_p = self.rbe()[1] #prob list for plane

        #data frame creation
        bird_plane = pd.DataFrame(list(zip(prob_list_b, prob_list_p, range(0,300))),
                         columns=['P(Bird|Obs)', 'P(Plane|Obs)', 'Time'])

        return bird_plane

    # def print_probs_nicely_plane(self):
    #     '''
    #     Purpose: To put all plane probabilties in a data frame
    #
    #     Arguments:
    #         pdf = a pandas dataframe containing the probability of being an airplane or bird based on speed
    #         data = a pandas dataframe containing the speed of 10 objects/tracks at 300 time points. (10 rows, 300 columns)
    #         obj = the object number (can be 0 to 9)
    #
    #     Returns:
    #         plane = dataframe of 2 columns: P(Plane) and Time
    #     '''
    #     prob_list_p = self.rbe()[1] #prob list for plane
    #
    #     #list of time points
    #     time = []
    #     for i in range(0,300):
    #         time.append(i)
    #
    #     #create data frame
    #     plane = pd.DataFrame(data = prob_list_p)
    #     plane['Time'] = time #add time column
    #     plane = plane.rename(columns={0: 'P(Plane)'}) #rename
    #
    #     return plane

################################################To run the code########################################################################################
#load data
pdf = pd.read_csv('pdf.txt', sep=",", header=None)
data = pd.read_csv('data.txt', sep=",", header=None)

#replace NaNs with mean spead of that object
for i in range(0,10):
    data.loc[i] = data.loc[i].fillna(value=data.loc[i].mean())


if __name__ == '__main__':
    for j in range(0,10): #to print dfs for each object/tack
        my_rbe = RBEWork(pdf, data, j)
        print("Probabilities for track/object:", j+1)
        print(my_rbe.print_probs_nicely())
        #print(my_rbe.print_probs_nicely_plane())
