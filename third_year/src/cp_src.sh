rm -rf ./wandb/
exp="../exp_results/$(date '+%Y_%m_%d_%H_%M_%S')"
mkdir $exp
cp -r "../src" $exp
cd $exp
mkdir ./results

echo $exp
