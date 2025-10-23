from typing import List, Union
import numpy as np
from languagechange.models.change.metrics import GradedChange, APD, PRT, PJSD
import languagechange.logging
from languagechange.logging import logging

def ma(ts, k):
    """
        Computes the moving average of a timeseries.
        Args:
            ts (np.array) : a timeseries.
            k (int) : the window (k timesteps to the left and k to the right)
        Returns:
            the moving average of the timeseries (not including endpoints)
    """
    return np.convolve(ts, np.ones(2*k+1))[2*k:-2*k] / (2*k+1)


class TimeSeries:

    def __init__(self, embs:List[np.array]=None, series:np.array = None, change_metric=None, timeseries_type:str = None, k=1, time_labels : Union[np.array, List] = None, clustering_algorithm = None, distance_metric='cosine'):
        # Init from embeddings
        if embs is not None:
            self.compute_from_embeddings(embs, change_metric, timeseries_type, k=k, time_labels=time_labels, clustering_algorithm=clustering_algorithm, distance_metric=distance_metric)
        # Init from an already constructed timeseries
        elif series is not None:
            self.series = series
            if time_labels is not None:
                self.ts = time_labels[self.series]
        else:
            self.series = np.array([])

    def compute_from_embeddings(self, embs : List[np.array], change_metric : Union[str, object], timeseries_type : str, k=1, time_labels : Union[np.array, List] = None, clustering_algorithm = None, distance_metric : str = 'cosine'):
        """
            Args:
                embs ([np.array]): a list of embeddings, each element of the list contains embeddings from one time period.
                change_metric (str|object): the metric to use when comparing embeddings from different time periods (should be one of the classes in languagechange.models.change.metrics).
                timeseries_type (str): the kind of timeseries to construct. One of ['compare_to_first', 'compare_to_last', 'consecutive', 'moving_average'].
                time_labels (np.array|list): labels for the x axis of the timeseries.
                clustering_algorithm: the clustering algorithm if using PJSD as the change metric. E.g. one of the algorithms in scikit-learn, or languagechange.
                distance_metric (str): the distance metric to use when computing change scores.
            Returns:
                series (np.array): the final timeseries.
                ts (np.array): the time values/labels for each value in the final timeseries.
        """
        if type(change_metric) == str:
            try:
                change_metric = {'apd': APD(), 'prt': PRT(), 'pjsd': PJSD()}[change_metric.lower()]
            except:
                logging.error("Error: if 'change_metric' is a string it must be one of 'apd','prt' and 'pjsd'.")
                raise Exception
            
        if not isinstance(change_metric, GradedChange):
            logging.error("Error: if 'change_metric' is an object it must be an instance of GradedChange.")
            raise Exception
        
        if isinstance(change_metric, PJSD):
            compute_scores = lambda e1, e2 : change_metric.compute_scores(e1, e2, clustering_algorithm, distance_metric)
        else:
            compute_scores = lambda e1, e2 : change_metric.compute_scores(e1, e2, distance_metric)

        
        # Compare every time period with the first one
        if timeseries_type == "compare_to_first":
            series = np.array([compute_scores(embs[0],emb) for emb in embs[1:]])
            t_idx = np.array(range(1,len(embs)))

        # Compare every time period with the last one
        elif timeseries_type == "compare_to_last":
            series = np.array([compute_scores(emb,embs[-1]) for emb in embs[:-1]])
            t_idx = np.array(range(len(embs)-1))

        # Compare consecutive time periods
        elif timeseries_type == "consecutive":
            series = np.array([compute_scores(embs[i],embs[i+1]) for i in range(len(embs)-1)])
            t_idx = np.array(range(1, len(embs)))

        # Moving average
        elif timeseries_type == "moving_average":
            series = ma(np.array([compute_scores(embs[i],embs[i+1]) for i in range(len(embs)-1)]), k)
            t_idx = np.array(range(k+1,len(embs)-k))

        if time_labels is not None:
            ts = np.array(time_labels)[t_idx]
        else:
            ts = t_idx

        self.series = series
        self.ts = ts
        return series, ts