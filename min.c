#include<stdio.h>
#include<limits.h>
int main()
{
    int nM,nT;
printf("\n enter no. of machines and tasks:\n");
scanf("%d%d",&nM,&nT);
int minMin[nM][nT];
int tmp[nM][nT];
int makespan=0;
printf("\n FILL DATA \n");
for(int i=0;i<nM;i++)
{
    for(int j=0;j<nT;j++)
    {
        scanf("%d",&minMIn[i][j]);
        tmp[i][j]=minMin[i][j];
    }
}
printf("\n ORIGINAL DATA\n");
for(int i=0;i<nM;i++)
{
    for(int j=0;j<nT;j++)
        printf("%d",minMIn[i][j])
    print("\n")
}
int resultTask[nt];
int resultMachine[nT];
int resultTime[nT];
int ptr=-1;
while(ptr<nT-1)
{
    int minimum=INT_MAX;
    int pos=-1;
    int time[nt],machine[nt];
    for(int i=0;i<nT;i++)
    {
        
        for(int j=0;j<nM;j++)
        {
            if(minMin[i][j]<minimum)
            {
                minMin[i][j]=minimum;
                pos=j;
            }
        }
        time[j]=minimum;
        machine[j]=pos;
    }
    for(int i=0;i<nT;i++)
    {
        if(time[i]<minimum){
            minimum=time[i];
            pos=machine[i];
        }
    }

resultTask[++ptr]=pos;
resultMachine[ptr]=machine[pos];
resultTime[ptr]=tmp[machine[pos]],[pos]
if (minimum > makespan)
{
    makespan = minimum;
}
for(int i=0;i<nM;i++)
{
    for(int j=0;j<nT;j++)
    {
        if(j==resultTask[ptr])
        {
            minMin[i][j]=INT_MAX;
        }
        else if(i==resultMachine[ptr]&&minMin[i][j]!=INT_MAX)
        {
            minMin[i][j]+=minimum;
        }
        else
        {
            continue;
        }
    }
}
}
printf("\nscheduled tasks are\n");
for(int i=0;i<nT;i++)
{
    printf("\nTask "%d"runs on machine"%d"with time"%d"units",resultTask[i]+1,resultMachine[i]+1,resultTime[i])

}
printf("\nmakespan:%d",makespan);
return 0;


}