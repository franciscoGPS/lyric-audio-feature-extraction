from sklearn.base import BaseEstimator, ClassifierMixin

class AudioClassifier(BaseEstimator, ClassifierMixin):  
	def __init__(self, arg1, arg2, arg3, ..., argN):

    # print("Initializing classifier:\n")

    args, _, _, values = inspect.getargvalues(inspect.currentframe())
    values.pop("self")

    for arg, val in values.items():
        setattr(self, arg, val)
        # print("{} = {}".format(arg,val)


     def fit(self, X, y=None):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """



        return self

    def _meaning(self, x):
        # returns True/False according to fitted classifier
        # notice underscore on the beginning
        return( True if x >= self.treshold_ else False )

    def predict(self, X, y=None):
        try:
            getattr(self, "treshold_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        return([self._meaning(x) for x in X])

    def score(self, X, y=None):
        # counts number of values bigger than mean
        return(sum(self.predict(X)))