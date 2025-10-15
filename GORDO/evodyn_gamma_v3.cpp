//Libraries
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <stdio.h>
#include <time.h>                      
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <vector>
#include <fstream>
#include <sstream> //ADDED BY JORGE
#include <map>
//End Libraries

using namespace std;

#define FileNameSize 100 //Used for the size of all variables corresponding to filenames

//Pseudo dynamic memory is changed according to the parameters
#define MemorySize1 50000 
#define MemorySize2 500000
#define MemorySize3 5000000
#define MemorySize4 50000000
int MemoryValue=MemorySize3;

bool debug=false;//Debugging flag, should be false

gsl_rng *generator;//GSL machinery to create random values

struct population {
    int originalN; //Total number of (initial) individuals
    int currentN; //Total number of individuals currently in the population
    int bottle_generations; //Number of generations between bottleneckings
    double popIncrease; //Factor for each population increases each "generation"
    int maxN;//Maximum number of individuals in the population (if population reaches this value, bottlenecking occurs)
    int nclassesYFP, nclassesCFP; //number of fitness groups in YFP and CFP
    
    int *n_individuals_class_YFP, *n_individuals_class_CFP;//number of individuals in each fitness group
    double *fitness_class_YFP, *fitness_class_CFP;//fitness value of each fitness group
    int *n_mutations_class_YFP, *n_mutations_class_CFP;//number of mutations of each fitness group
    
    //Random distribution parameters
    double alpha, beta, Ub, tau;    
};

//Functions to read time
long int clo;
void startTimer(string location) {clo=clock(); cout<<"Timer started at "<<location<<"\n";}
void endTimer(){char taken[100]; float diff=((float)clock() - (float)clo) / 1000000.0F; sprintf(taken, "Time taken : %4.8f\n", diff); cout<<taken;}
//End of functions to read time

void InitializePopulation(population *pop)
{
    //Initial population numbers
    pop->currentN=pop->originalN;
    
    //Initialize structures
    pop->fitness_class_YFP = new double[MemoryValue];
    pop->fitness_class_CFP = new double[MemoryValue];
    pop->n_individuals_class_YFP = new int[MemoryValue];
    pop->n_individuals_class_CFP = new int[MemoryValue];
    
    pop->n_mutations_class_YFP = new int[MemoryValue];
    pop->n_mutations_class_CFP = new int[MemoryValue];
    
    //Initialize first fitness groups
    pop->fitness_class_YFP[0] = 1.; //Initial fitness is 1
    pop->fitness_class_CFP[0] = 1.; //Initial fitness is 1
    pop->n_individuals_class_YFP[0] = pop->originalN/2; //Initial population is half for each subpopulation (proportional splitting)
    pop->n_individuals_class_CFP[0] = pop->originalN/2; //Initial population is half for each subpopulation (proportional splitting)
    pop->nclassesYFP = 1; //One fitness class at the beginning for YFP
    pop->nclassesCFP = 1; //One fitness class at the beginning for CFP
    pop->n_mutations_class_YFP[0] = 0; //No initial mutations (ancestral)
    pop->n_mutations_class_CFP[0] = 0; //No initial mutations (ancestral)   
}

void Reproduction(population *pop, bool do_I_bottle)
{   
    bool doubleBottleneck=false;
    
    unsigned int *next_generation_distribution; //Holds the distribution (both from YFP - top half - and CFP - lower half) given by multinomial, represents the numbers after "bottlenecking"
    double *fitness_fractions; //The fitness of each class (both from YFP and CFP) relative to total fitness of the population
    int *to_be_mutated_YFP, *to_be_mutated_CFP; //For each fitness class, holds how many of its individuals are going to be mutated
    
    int totalclasses=pop->nclassesYFP+pop->nclassesCFP;
    
    //Memory allocation (will not be higher than the total number of classes)
    next_generation_distribution = new unsigned int[totalclasses];
    fitness_fractions = new double[totalclasses];
    to_be_mutated_YFP = new int[totalclasses];
    to_be_mutated_CFP = new int[totalclasses];
    
    double TotalFitness=0;
    
    //STEP 1: Calculate total fitness of the population
    for(int fitnessgroupYFP=0; fitnessgroupYFP<pop->nclassesYFP; fitnessgroupYFP++)
        TotalFitness += pop->fitness_class_YFP[fitnessgroupYFP]*pop->n_individuals_class_YFP[fitnessgroupYFP];
    
    for(int fitnessgroupCFP=0; fitnessgroupCFP<pop->nclassesCFP; fitnessgroupCFP++)
        TotalFitness += pop->fitness_class_CFP[fitnessgroupCFP]*pop->n_individuals_class_CFP[fitnessgroupCFP];
    
    
    //STEP 2: Calculate fitness of each fitness group relative to the entire population (total fitness)
    for(int fitnessgroupYFP=0; fitnessgroupYFP<pop->nclassesYFP; fitnessgroupYFP++)
    {
        double relative_fitness=pop->fitness_class_YFP[fitnessgroupYFP]*pop->n_individuals_class_YFP[fitnessgroupYFP]; //Combined fitness: n_individuals x fitness value
        fitness_fractions[fitnessgroupYFP]=relative_fitness/TotalFitness;
    }
    
    for(int fitnessgroupCFP=pop->nclassesYFP; fitnessgroupCFP<totalclasses; fitnessgroupCFP++)
    {
        double relative_fitness=pop->fitness_class_CFP[fitnessgroupCFP-pop->nclassesYFP]*pop->n_individuals_class_CFP[fitnessgroupCFP-pop->nclassesYFP];
        fitness_fractions[fitnessgroupCFP]=relative_fitness/TotalFitness;
    }
    
    //STEP 3: Bottlenecking: assert how many individuals from each fitness class will "survive" to the next "generation". Probability of "surviving" is proportional to the fitness value of the fitness group
    gsl_ran_multinomial(generator,totalclasses,pop->currentN,fitness_fractions,next_generation_distribution);
        
    //STEP 4: Reassign new population numbers
    for(int fitnessgroupYFP=0; fitnessgroupYFP<pop->nclassesYFP; fitnessgroupYFP++ )
        pop->n_individuals_class_YFP[fitnessgroupYFP]=next_generation_distribution[fitnessgroupYFP];
    
    for(int fitnessgroupCFP=pop->nclassesYFP; fitnessgroupCFP<totalclasses; fitnessgroupCFP++ )
        pop->n_individuals_class_CFP[fitnessgroupCFP-pop->nclassesYFP]=next_generation_distribution[fitnessgroupCFP];
    
    //STEP 5: Calculate, using the probability given by the mutation rate (Ub*tau for YFP, Ub for CFP), how many individuals from each fitness class will undergo mutation. Because their fitness will be changed, we can already remove them from their original fitness groups (they will create new fitness groups)
    for(int fitnessgroupYFP=0; fitnessgroupYFP<pop->nclassesYFP; fitnessgroupYFP++ )
    {
        to_be_mutated_YFP[fitnessgroupYFP] = gsl_ran_binomial(generator, pop->Ub*pop->tau, pop->n_individuals_class_YFP[fitnessgroupYFP]);
        pop->n_individuals_class_YFP[fitnessgroupYFP]-=to_be_mutated_YFP[fitnessgroupYFP];
    }
    
    for(int fitnessgroupCFP=0; fitnessgroupCFP<pop->nclassesCFP; fitnessgroupCFP++ )
    {
        to_be_mutated_CFP[fitnessgroupCFP] = gsl_ran_binomial(generator, pop->Ub, pop->n_individuals_class_CFP[fitnessgroupCFP]);
        pop->n_individuals_class_CFP[fitnessgroupCFP]-=to_be_mutated_CFP[fitnessgroupCFP];
    }
    
    //STEP 6: Create the new fitness groups that originate from each of the mutated individuals
    int new_class_counter_YFP = 0;
    for (int fitnessgroupYFP=0; fitnessgroupYFP<pop->nclassesYFP; fitnessgroupYFP++)
        for(int individual=0; individual<to_be_mutated_YFP[fitnessgroupYFP]; individual++)//For each new individual to be originated from each fitnessgroupYFP
        {
            int newclass=pop->nclassesYFP+new_class_counter_YFP; //New class index
            pop->n_individuals_class_YFP[newclass]=1; //Has 1 individual (the mutation one)
        
            pop->fitness_class_YFP[newclass]=pop->fitness_class_YFP[fitnessgroupYFP]*(1+gsl_ran_gamma(generator, pop->alpha, pop->beta)); //New fitness value, is the sum of the previous fitness value plus a small value given by the gamma distribution. Mutations are ALWAYS benefitial
            
            //cout<<"From -> "<<(pop->fitness_class_YFP[newclass]-pop->fitness_class_YFP[fitnessgroupYFP])<<"\n";
            
            pop->n_mutations_class_YFP[newclass]=pop->n_mutations_class_YFP[fitnessgroupYFP]+1; //Increase number of mutations
            new_class_counter_YFP++; //Increase number of new fitness classes for YFP
        }
    pop->nclassesYFP+=new_class_counter_YFP; //Update new number of fitness classes for YFP
    
    int new_class_counter_CFP = 0;
    for (int fitnessgroupCFP=0; fitnessgroupCFP<pop->nclassesCFP; fitnessgroupCFP++)
        for(int individual=0; individual<to_be_mutated_CFP[fitnessgroupCFP]; individual++)//For each new individual to be originated from each fitnessgroupCFP
        {
            int newclass=pop->nclassesCFP+new_class_counter_CFP; //New class index
            pop->n_individuals_class_CFP[newclass]=1; //Has 1 individual (the mutation one)
            pop->fitness_class_CFP[newclass]=pop->fitness_class_CFP[fitnessgroupCFP]*(1+gsl_ran_gamma(generator, pop->alpha, pop->beta)); //New fitness value, is the sum of the previous fitness value plus a small value given by the gamma distribution. Mutations are ALWAYS benefitial
            pop->n_mutations_class_CFP[newclass]=pop->n_mutations_class_CFP[fitnessgroupCFP]+1; //Increase number of mutations
            new_class_counter_CFP++; //Increase number of new fitness classes for CFP
        }
    pop->nclassesCFP+=new_class_counter_CFP; //Update new number of fitness classes for CFP
    
    //STEP 7: Check which classes ended up with 0 individuals (all of them created new classes due to mutations), and shift the arrays accordingly
    int count_YFP_zero_individuals=0;
    for (int fitnessgroupYFP=0; fitnessgroupYFP<pop->nclassesYFP; fitnessgroupYFP++)
    {
        if (pop->n_individuals_class_YFP[fitnessgroupYFP]==0) count_YFP_zero_individuals++; //Count how many "contiguous" classes have 0 individuals
        else //Remove all classes that was counted in the previous step, by shifting all the classes ahead "count_YFP_zero_individuals" times
        {
            pop->n_individuals_class_YFP[fitnessgroupYFP-count_YFP_zero_individuals]=pop->n_individuals_class_YFP[fitnessgroupYFP];
            pop->fitness_class_YFP[fitnessgroupYFP-count_YFP_zero_individuals]=pop->fitness_class_YFP[fitnessgroupYFP];
            pop->n_mutations_class_YFP[fitnessgroupYFP-count_YFP_zero_individuals]=pop->n_mutations_class_YFP[fitnessgroupYFP];
        }
    }
    pop->nclassesYFP-=count_YFP_zero_individuals;
    
    int count_CFP_zero_individuals=0;
    for (int fitnessgroupCFP=0; fitnessgroupCFP<pop->nclassesCFP; fitnessgroupCFP++)
    {
        if (pop->n_individuals_class_CFP[fitnessgroupCFP]==0) count_CFP_zero_individuals++;//Count how many "contiguous" classes have 0 individuals
        else //Remove all classes that was counted in the previous step, by shifting all the classes ahead "count_YFP_zero_individuals" times
        {
            pop->n_individuals_class_CFP[fitnessgroupCFP-count_CFP_zero_individuals]=pop->n_individuals_class_CFP[fitnessgroupCFP];
            pop->fitness_class_CFP[fitnessgroupCFP-count_CFP_zero_individuals]=pop->fitness_class_CFP[fitnessgroupCFP];
            pop->n_mutations_class_CFP[fitnessgroupCFP-count_CFP_zero_individuals]=pop->n_mutations_class_CFP[fitnessgroupCFP];
        }
    }
    pop->nclassesCFP-=count_CFP_zero_individuals;
    
    //Cleanup memory
    delete[] to_be_mutated_YFP;
    delete[] to_be_mutated_CFP;
    delete[] next_generation_distribution;
    delete[] fitness_fractions;
    
        
    if(do_I_bottle)//Additional "bottleneck", or the "true bottleneck"
    {
        
        //cout<<"BOTTLENECKING!!\n";
        //cout<<"Population reached "<<pop->currentN<<" individuals, time for bottlenecking\n";
        
        int totalclasses=pop->nclassesYFP+pop->nclassesCFP;
        
        //Memory allocation (will not be higher than the total number of classes)
        next_generation_distribution = new unsigned int[totalclasses];
        fitness_fractions = new double[totalclasses];
        
        double TotalFitness=0;
        
        //STEP 1: Calculate total fitness of the population
        for(int fitnessgroupYFP=0; fitnessgroupYFP<pop->nclassesYFP; fitnessgroupYFP++)
            TotalFitness += pop->fitness_class_YFP[fitnessgroupYFP]*pop->n_individuals_class_YFP[fitnessgroupYFP];
        
        for(int fitnessgroupCFP=0; fitnessgroupCFP<pop->nclassesCFP; fitnessgroupCFP++)
            TotalFitness += pop->fitness_class_CFP[fitnessgroupCFP]*pop->n_individuals_class_CFP[fitnessgroupCFP];
        
        
        //STEP 2: Calculate fitness of each fitness group relative to the entire population (total fitness)
        for(int fitnessgroupYFP=0; fitnessgroupYFP<pop->nclassesYFP; fitnessgroupYFP++)
        {
            double relative_fitness=pop->fitness_class_YFP[fitnessgroupYFP]*pop->n_individuals_class_YFP[fitnessgroupYFP]; //Combined fitness: n_individuals x fitness value
            fitness_fractions[fitnessgroupYFP]=relative_fitness/TotalFitness;
        }
        
        for(int fitnessgroupCFP=pop->nclassesYFP; fitnessgroupCFP<totalclasses; fitnessgroupCFP++)
        {
            double relative_fitness=pop->fitness_class_CFP[fitnessgroupCFP-pop->nclassesYFP]*pop->n_individuals_class_CFP[fitnessgroupCFP-pop->nclassesYFP];
            fitness_fractions[fitnessgroupCFP]=relative_fitness/TotalFitness;
        }
        
        pop->currentN=pop->originalN;
        
        //STEP 3: Bottlenecking: assert how many individuals from each fitness class will "survive" to the next "generation". Probability of "surviving" is proportional to the fitness value of the fitness group
        gsl_ran_multinomial(generator,totalclasses,pop->currentN,fitness_fractions,next_generation_distribution);
        
        //STEP 4: Reassign new population numbers
        for(int fitnessgroupYFP=0; fitnessgroupYFP<pop->nclassesYFP; fitnessgroupYFP++ )
            pop->n_individuals_class_YFP[fitnessgroupYFP]=next_generation_distribution[fitnessgroupYFP];
        
        for(int fitnessgroupCFP=pop->nclassesYFP; fitnessgroupCFP<totalclasses; fitnessgroupCFP++ )
            pop->n_individuals_class_CFP[fitnessgroupCFP-pop->nclassesYFP]=next_generation_distribution[fitnessgroupCFP];
        
        //STEP 5 & 6: NO MUTATION OCCURS IN BOTTLENECKING
        
        //STEP 7: Check which classes ended up with 0 individuals (all of them created new classes due to mutations), and shift the arrays accordingly
        int count_YFP_zero_individuals=0;
        for (int fitnessgroupYFP=0; fitnessgroupYFP<pop->nclassesYFP; fitnessgroupYFP++)
        {
            if (pop->n_individuals_class_YFP[fitnessgroupYFP]==0) count_YFP_zero_individuals++; //Count how many "contiguous" classes have 0 individuals
            else //Remove all classes that was counted in the previous step, by shifting all the classes ahead "count_YFP_zero_individuals" times
            {
                pop->n_individuals_class_YFP[fitnessgroupYFP-count_YFP_zero_individuals]=pop->n_individuals_class_YFP[fitnessgroupYFP];
                pop->fitness_class_YFP[fitnessgroupYFP-count_YFP_zero_individuals]=pop->fitness_class_YFP[fitnessgroupYFP];
                pop->n_mutations_class_YFP[fitnessgroupYFP-count_YFP_zero_individuals]=pop->n_mutations_class_YFP[fitnessgroupYFP];
            }
        }
        pop->nclassesYFP-=count_YFP_zero_individuals;
        
        int count_CFP_zero_individuals=0;
        for (int fitnessgroupCFP=0; fitnessgroupCFP<pop->nclassesCFP; fitnessgroupCFP++)
        {
            if (pop->n_individuals_class_CFP[fitnessgroupCFP]==0) count_CFP_zero_individuals++;//Count how many "contiguous" classes have 0 individuals
            else //Remove all classes that was counted in the previous step, by shifting all the classes ahead "count_YFP_zero_individuals" times
            {
                pop->n_individuals_class_CFP[fitnessgroupCFP-count_CFP_zero_individuals]=pop->n_individuals_class_CFP[fitnessgroupCFP];
                pop->fitness_class_CFP[fitnessgroupCFP-count_CFP_zero_individuals]=pop->fitness_class_CFP[fitnessgroupCFP];
                pop->n_mutations_class_CFP[fitnessgroupCFP-count_CFP_zero_individuals]=pop->n_mutations_class_CFP[fitnessgroupCFP];
            }
        }
        pop->nclassesCFP-=count_CFP_zero_individuals;
        
        delete[] next_generation_distribution;
        delete[] fitness_fractions;
    }
    else
    {
        //ADDITIONAL STEP - UPDATE POPULATION NUMBERS: Calculate the updated N of the population
        if(pop->originalN!=pop->maxN)
            if(pop->currentN<pop->maxN)
                pop->currentN*=pop->popIncrease;
    }
}

//Calculates the YFP/(YFP+CFP) ratio of the population
//void GetYFPRatio(population *pop, double &YFPratio, double &TotFitness, int &sumYFP, int &sumCFP)
void GetYFPFrequency_and_Ratio(population *pop, double &YFPfreq, double &YFPratio, double &TotFitness, int &sumYFP, int &sumCFP)
{
    double TotalFitness=0;
    
    //Calculates number of individuals in YFP group (and first half of the TotalFitness)
    int sum_YFP_individuals=0;
    for(int fitnessgroupYFP=0; fitnessgroupYFP<pop->nclassesYFP; fitnessgroupYFP++)
    {
        TotalFitness+=pop->fitness_class_YFP[fitnessgroupYFP]*pop->n_individuals_class_YFP[fitnessgroupYFP];
        sum_YFP_individuals+=pop->n_individuals_class_YFP[fitnessgroupYFP];
    }
    
    //Calculates number of individuals in CFP group (and last half of the TotalFitness)
    int sum_CFP_individuals=0;
    for(int fitnessgroupCFP=0; fitnessgroupCFP<pop->nclassesCFP; fitnessgroupCFP++)
    {
        TotalFitness+=pop->fitness_class_CFP[fitnessgroupCFP]*pop->n_individuals_class_CFP[fitnessgroupCFP];
        sum_CFP_individuals+=pop->n_individuals_class_CFP[fitnessgroupCFP];
    }
    
    //These assignments will all go to the "outside", where the function was called
    YFPfreq=(double)(sum_YFP_individuals)/(sum_CFP_individuals+sum_YFP_individuals);     //Calculates FREQUENCY
    YFPratio=(double)(sum_YFP_individuals)/(sum_CFP_individuals);     //Calculates RATIO
        
    TotFitness=TotalFitness;
    sumYFP=sum_YFP_individuals;
    sumCFP=sum_CFP_individuals;
}


//Variance
double FitnessVariance(double *sampleFitness, int samplesize)
{
    double var1=0, var2=0;
    for (int pick=0; pick<samplesize; pick++)
    {

        var1+=sampleFitness[pick];
        var2+=pow(sampleFitness[pick],2);
    }
    
    var1=pow((var1/samplesize),2);
    var2=var2/samplesize;

    return fabs(var2-var1);
}

void GetFitnessSample_WholePop(population *pop, int sum_CFP_individuals, int sum_YFP_individuals, double& fitness_average_dominant, double& fitness_average_whole)
{
    int samplingsize=sum_CFP_individuals+sum_YFP_individuals;
    double picked_fitnesses[samplingsize];
    
    double total_fitness_sum_CFP=0;
    double total_fitness_sum_YFP=0;
    
    for (int class_CFP=0; class_CFP<pop->nclassesCFP; class_CFP++)
        total_fitness_sum_CFP+=pop->n_individuals_class_CFP[class_CFP]*pop->fitness_class_CFP[class_CFP];
    
    for (int class_YFP=0; class_YFP<pop->nclassesYFP; class_YFP++)
        total_fitness_sum_YFP+=pop->n_individuals_class_YFP[class_YFP]*pop->fitness_class_YFP[class_YFP];
    
    
    if(sum_CFP_individuals>sum_YFP_individuals)
        fitness_average_dominant=total_fitness_sum_CFP/sum_CFP_individuals;
    
    if(sum_CFP_individuals<sum_YFP_individuals)
        fitness_average_dominant=total_fitness_sum_YFP/sum_YFP_individuals;    
    
    fitness_average_whole=(total_fitness_sum_CFP+total_fitness_sum_YFP)/(sum_CFP_individuals+sum_YFP_individuals);
}

void GetFitnessSample(population *pop, int samplingsize, int sum_CFP_individuals, int sum_YFP_individuals, double& fitness_dominant_group, double& variance_fitness)
{
    double picked_fitnesses[samplingsize];

    if(sum_CFP_individuals>sum_YFP_individuals) //CFP is higher represented
    {
        unsigned int distributionCFP[pop->nclassesCFP];                                        
        double number_fractions_CFP[pop->nclassesCFP];
        
        //Creates relative fractions between each CFP class number of individuals and total number of CFP individuals (all classes)
        for (int cls=0; cls<pop->nclassesCFP; cls++)
            number_fractions_CFP[cls]=((double)pop->n_individuals_class_CFP[cls]/(double)sum_CFP_individuals);
        
        //Runs multinomial, with "samplingsize" individuals chosen (equivalent to experimentally picking a number of them at random)
        gsl_ran_multinomial(generator,pop->nclassesCFP,samplingsize,number_fractions_CFP,distributionCFP);
        
        int picked=0;
        
        for (int cls=0; cls<pop->nclassesCFP; cls++)
            for (int p=0; p<distributionCFP[cls]; p++)
                picked_fitnesses[picked++]=pop->fitness_class_CFP[cls];
    }
    
    if(sum_CFP_individuals<sum_YFP_individuals) //YFP is higher represented
    {
        unsigned int distributionYFP[pop->nclassesYFP];                                        
        double number_fractions_YFP[pop->nclassesYFP];
        
        //Creates relative fractions between each YFP class number of individuals and total number of CFP individuals (all classes)
        for (int cls=0; cls<pop->nclassesYFP; cls++)
            number_fractions_YFP[cls]=((double)pop->n_individuals_class_YFP[cls]/(double)sum_YFP_individuals);
        
        //Runs multinomial, with "samplingsize" individuals chosen (equivalent to experimentally picking a number of them at random)
        gsl_ran_multinomial(generator,pop->nclassesYFP,samplingsize,number_fractions_YFP,distributionYFP);
        
        int picked=0;
        for (int cls=0; cls<pop->nclassesYFP; cls++)
            for (int p=0; p<distributionYFP[cls]; p++)
                picked_fitnesses[picked++]=pop->fitness_class_YFP[cls];
    }
    
    //Averages fitness of samples chosen
    double avfitness=0;
    for (int p=0; p<samplingsize; p++){/*cout<<"Fitness->"<<picked_fitnesses[p]<<"\n";*/ avfitness+=picked_fitnesses[p];}
    fitness_dominant_group=avfitness/samplingsize;
    
    //Gets Variance for fitness samples
    variance_fitness=FitnessVariance(picked_fitnesses, samplingsize);
}

map<int, string > CreateStatisticsIndex()
{
    map<int, string> statistics_index;

    int curr_index=0;

    //Fitness Related
    statistics_index[curr_index++]="Mid Fitness Average";
    statistics_index[curr_index++]="Mid Fitness Variance";
    statistics_index[curr_index++]="Final Fitness Average";
    statistics_index[curr_index++]="Final Fitness Variance";
    
    //Frequency Related
    statistics_index[curr_index++]="Mid Frequency";
    statistics_index[curr_index++]="Final Frequency";
    statistics_index[curr_index++]="Average Frequency";
    
    //Deviation Thresholds
    statistics_index[curr_index++]="Bottleneck of Threshold Deviation";
    statistics_index[curr_index++]="Slope at Deviation";
    statistics_index[curr_index++]="Maximum Deviation Registered";
    statistics_index[curr_index++]="Bottlenecks Above Threshold Deviation";
    
    //Reversals
    statistics_index[curr_index++]="Number of Reversals";
    
    //Fixation
    statistics_index[curr_index++]="Bottlenecks until Fixation";
    statistics_index[curr_index++]="Bottlenecks of Fixation";
    
    return statistics_index;
}


//Added version 2.1
double unifRand()
{
    return rand() / double(RAND_MAX);
}

double unifRand(double a, double b)
{
    return (b-a)*unifRand() + a;
}
//END Added version 2.1

int main(int ac, char **av)
{
    
    population popul; //Main variable, holds entire population elements and parameters
    
    int generation, max_generations, simulation, simulation_repeats;
    double ratio_deviation; //threshold for ratio deviation
    
    // cout<<ac<<"\n";
    
    if (ac!=12)
    {
        cout <<  "WRONG ARGUMENTS! :\n"<< " <N0> <NMax> <BottleneckGenerations> <Log Interval> <Max Generations> <Simulation Repeats> <Ub> <tau> <alpha> <beta> <random_seed_gsl>\n";
        return -1;
    }
    
    int j=0;
    
    
    popul.originalN = atoi (av[++j]);
    popul.maxN = atoi (av[++j]);
    popul.bottle_generations = atoi (av[++j]);
    int log_interval = atoi (av[++j]);
    max_generations = atoi (av[++j]);
    simulation_repeats = atoi (av[++j]);
    popul.Ub=atof (av[++j]);
    popul.tau=atof (av[++j]);
    popul.alpha=atof (av[++j]);
    popul.beta=atof (av[++j]);
    
    //Change version 2.1
    unsigned int gsl_seed = atoi(av[++j]);
    
    cout << "# invocation:\n\t";
    for (int i=0; i<ac; i++)
        cout << av[i] << " ";
    cout << endl;
    
    map<int, string> statistics_index=CreateStatisticsIndex();
    map<string, vector<double>  > statistics;
    
    FILE *DistrStatistics; //Pointer to output files    
    char arq_SS2[100];//Stores name of the output files
    sprintf(arq_SS2,"SummaryStatistics_all_SINGLE.txt");
    
    FILE *SumStatistics; //Pointer to output files    
    char arq_SS[100];//Stores name of the output files
    
    //srand(unif_seed);
    
    //Moved+Added in 2.4
    //int total_bottlenecks=max_generations/(popul.bottle_generations);
    int total_number_bottlenecks=max_generations/popul.bottle_generations;
    double whole_pop_fitness_average_b[simulation_repeats][total_number_bottlenecks+2];
    double frequencies_b[simulation_repeats][total_number_bottlenecks+2];
    
    //Initialize
    for (int a=0; a<simulation_repeats; a++) {
        for (int b=0; b<total_number_bottlenecks+2; b++)
        {
            whole_pop_fitness_average_b[a][b]=-100;
            frequencies_b[a][b]=-100;
        }
    }
     
    //popul.Ub=pow(10, unifRand(-4,-9));
    //popul.alpha=unifRand(0.5,15);
    //popul.beta=pow(10, unifRand(-0.08,-4));

    cout<<"# Evolutionary parameters:\n";
    cout<<"\tMutation rate "<<popul.Ub<<"\n";
    cout<<"\tMutation rate increase "<<popul.tau<<"\n";
    cout<<"\tShape (Alpha) "<<popul.alpha<<"\n";
    cout<<"\tScale (Beta) "<<popul.beta<<"\n";
    
    if(popul.originalN==popul.maxN) popul.popIncrease=1; else popul.popIncrease=2;//NEW!
    
    generator=gsl_rng_alloc(gsl_rng_mt19937);// initializing random number
    gsl_rng_set (generator, gsl_seed); //Define a seed for random values 
    
      
    //Main Cycle, will repeat evolution "simulation_repeats" times
    for (simulation=1; simulation<=simulation_repeats; simulation++)
    {        
        InitializePopulation(&popul);//Initializes population structures (arrays) and creates initial fitness classes, individuals, fitness values and mutations
        
        int bottleneck_timings=0;
        int current_bottleneck=0;
        
        //Repeats reproductory cycle for "max_generations"
        for (generation=0; generation<max_generations; generation++)
        {
            

            //cout<<"Generation -> "<<generation<<"   "<<popul.currentN<<" individuals \n";
            

             if (generation==0)//IF first generation, fill up initial frequencies (should be 50%)
             {
                 double  YFPfreq, YFPratio, TotalFitness;
                 int sum_YFP_individuals, sum_CFP_individuals;
                 GetYFPFrequency_and_Ratio(&popul, YFPfreq, YFPratio, TotalFitness, sum_YFP_individuals, sum_CFP_individuals);
                 frequencies_b[simulation-1][0]=YFPfreq;
             }
            
            //Bottleneck control (external to reproduction function)
            double do_I_bottle=false;
            
            if(bottleneck_timings==popul.bottle_generations-1){do_I_bottle=true; bottleneck_timings=0; current_bottleneck++;} //If time (generations) passed since last bottleneck is the specified bottleneck time...
            else bottleneck_timings++;//If not, keep counting
            
            Reproduction(&popul, do_I_bottle); //Go through the whole reproduction cycle
            
            //LOG STATISTICS (always after reproduction)//
            

             //###########   RATIOS  ###########
             //Asserts ratio of YFP population, along with total fitness of the entire population and the number of YFP and CFP individuals
             double YFPfreq, YFPratio, TotalFitness;
             int sum_YFP_individuals, sum_CFP_individuals;
             GetYFPFrequency_and_Ratio(&popul, YFPfreq, YFPratio, TotalFitness, sum_YFP_individuals, sum_CFP_individuals);
             
                             
             if(do_I_bottle)
             {
                 //cout<<"Bottlenecked at generation "<<generation<<", it was bottlenec #"<<current_bottleneck<<" current size is "<<popul.currentN<<"\n";
                 
                 frequencies_b[simulation-1][current_bottleneck]=YFPfreq;
                 
                 if(current_bottleneck%log_interval==0)
                 {
                 double fit_average_dom, fit_average_whole;
                 //Fitness of whole population and dominant group population
                 GetFitnessSample_WholePop(&popul, sum_CFP_individuals, sum_YFP_individuals, fit_average_dom, fit_average_whole);
                 whole_pop_fitness_average_b[simulation-1][current_bottleneck]=fit_average_whole;

             }
             
             
            }
             
             //Get statistics and the end of the run (Use only info stored in bottlenecks structures)
             if (generation==max_generations-1)
             {
             
                 GetYFPFrequency_and_Ratio(&popul, YFPfreq, YFPratio, TotalFitness, sum_YFP_individuals, sum_CFP_individuals);                         
                 frequencies_b[simulation-1][current_bottleneck+1]=YFPfreq;//Because evolutionary run may end before bottleneck, add last value
            }

        }
        
        //Cleanup previous population arrays
        delete [] popul.n_individuals_class_YFP;
        delete [] popul.n_individuals_class_CFP;
        delete [] popul.fitness_class_YFP;
        delete [] popul.fitness_class_CFP;
        delete [] popul.n_mutations_class_YFP;
        delete [] popul.n_mutations_class_CFP;
    }
    
    DistrStatistics = fopen(arq_SS2,"a");
    fprintf(DistrStatistics,"->[Unif Seed:NA] %g %g %g\n", popul.Ub, popul.alpha, popul.beta);
    fclose(DistrStatistics);
    
    sprintf(arq_SS,"output/SummaryStatistics_Ub_%g_tau_%g_alpha_%g_beta_%g.txt", popul.Ub, popul.tau, popul.alpha, popul.beta);
    cout<<"Output file "<<arq_SS<<"\n";
    SumStatistics = fopen(arq_SS,"w");
    
    fprintf(SumStatistics,"Population");
    
    for (int ind=log_interval; ind<total_number_bottlenecks+2; ind+=log_interval)
        fprintf(SumStatistics,"\t Frequency at Bottleneck %d", ind);
    
    for (int ind=log_interval; ind<total_number_bottlenecks+2; ind+=log_interval)
        fprintf(SumStatistics,"\t Whole Population Fitness Average at Bottleneck %d", ind);
    
    fprintf(SumStatistics,"\n");    
    

    for (int sim=0; sim<simulation_repeats; sim++)
    {
        fprintf(SumStatistics,"POP-%d ", sim+1);
        
        for (int ind=log_interval; ind<total_number_bottlenecks+2; ind+=log_interval)
            fprintf(SumStatistics,"\t %g ", frequencies_b[sim][ind]);

        for (int ind=log_interval; ind<total_number_bottlenecks+2; ind+=log_interval)
            fprintf(SumStatistics,"\t %g ", whole_pop_fitness_average_b[sim][ind]);

        
        fprintf(SumStatistics,"\n");
    }
    
    fclose(SumStatistics);
    
    //cout<<"Resetting matrixes...\n";
    for (int a=0; a<simulation_repeats; a++) {
        for (int b=0; b<total_number_bottlenecks+2; b++)
        {
            whole_pop_fitness_average_b[a][b]=-100;
            frequencies_b[a][b]=-100;
        }
    }
    return 0;

}


