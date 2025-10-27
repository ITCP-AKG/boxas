import numpy as np
import operator
import traceback

import optuna

def suggest_type(trial, name, typ, bounds):
    suggestors = {
        'int': lambda: trial.suggest_int(name, *bounds),
        'float': lambda: trial.suggest_float(name, *bounds),
        'categorical': lambda: trial.suggest_categorical(name, *bounds),
        'uniform': lambda: trial.suggest_uniform(name, *bounds),
        'loguniform': lambda: trial.suggest_loguniform(name, *bounds),
        'choice': lambda: trial.suggest_categorical(name, *bounds)
    }

    return suggestors.get(typ, lambda: trial.suggest_float(name, *bounds))()

class XASOpt(object):
    def __init__(self, study_name, hyperparameters, template, compounds, runner, loss, tool='feff'):
        self._study_name = study_name
        self._hyperparameters = hyperparameters
        self._runner = runner
        self._compounds = compounds
        self._template = template
        self._loss = loss
        self._tool = tool

        self._storage = "mysql+pymysql://optuna:optuna_pw@localhost/optuna_db"
        
    def objective(self, trial):
        trial_pars = {}
        for name, par in self._hyperparameters.items():
            trial_pars[name] = suggest_type(trial, name, par['type'], par['bounds'])

        X = self._runner(self._tool, self._study_name, self._template, self._compounds, trial_pars, trial.number)

        try:
            loss_value, energy_scale = self._loss.calculate(X)
            
            # Report intermediate value for pruning
            trial.report(loss_value, step=1)

            trial.set_user_attr("custom_info", {f"energy_scale_{i}": eo for i, eo in enumerate(energy_scale)})
                
        except Exception as e:
            print(f"Error calculating loss for trial {trial.number}: {e}")
            print("Full traceback:")
            traceback.print_exc()
            raise optuna.TrialPruned()

        return loss_value

    def optimize(self, n_trials=10, load_if_exists=False, 
             patience=None, random_search=False):
        """
        Run the optimization process with optional early stopping.
        
        Args:
            n_trials (int): Number of trials to run
            n_jobs (int): Number of parallel jobs
            study_name (str, optional): Name for the study in the database. 
                                   If None, self._study_name will be used.
            load_if_exists (bool): If True, load existing study with the same name
            patience (int, optional): Number of trials to wait for improvement before stopping.
                                If None, no early stopping is applied.
    
        Returns:
            dict: Best parameters found during optimization
        """
        
        # Create the study
        direction="minimize"
        if random_search:
            sampler = optuna.samplers.RandomSampler() 
        else:
            sampler = optuna.samplers.TPESampler() 
            # sampler = optuna.samplers.GPSampler() 

        study = optuna.create_study(
            storage=self._storage,
            sampler=sampler,
            study_name=self._study_name,
            direction=direction,
            load_if_exists=load_if_exists,
        )
        
        # Add early stopping callback if patience is specified
        callbacks = []
        if patience is not None:
            early_stopping = EarlyStoppingCallback(patience, direction=direction)
            callbacks.append(early_stopping)
        
        # Run optimization with callbacks
        study.optimize(
            self.objective, 
            n_trials=n_trials, 
            callbacks=callbacks,
            n_jobs=5 if random_search else 1,
        )
        
        best_trial = study.best_trial
        best_params = best_trial.params
        best_value = best_trial.value
        
        print(f"Best trial of {self._study_name}: {best_trial.number}")
        print(f"Best value: {best_value}")
        print(f"Best parameters: {best_params}")
        
        return best_params

class EarlyStoppingCallback(object):
    """Early stopping callback for Optuna."""

    def __init__(self, early_stopping_rounds: int, direction: str = "minimize") -> None:
        self.early_stopping_rounds = early_stopping_rounds

        self._iter = 0

        if direction == "minimize":
            self._operator = operator.lt
            self._score = np.inf
        elif direction == "maximize":
            self._operator = operator.gt
            self._score = -np.inf
        else:
            ValueError(f"invalid direction: {direction}")

    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        # if len(study.trials) <= 10:
        #     return  # Early stopping not needed for less than 20 trials.

        """Do early stopping."""
        if self._operator(study.best_value, self._score):
            self._iter = 0
            self._score = study.best_value
        else:
            self._iter += 1

        if self._iter >= self.early_stopping_rounds:
            study.stop()

