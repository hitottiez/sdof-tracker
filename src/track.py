import numpy as np
import statistics
import scipy.stats as stats
from scipy.spatial import distance
import pandas as pd

def calc_mu_cov(x):
    mu = x.mean(axis=0)
    cov = np.cov(x.T)

    return mu, cov


# Hotelling method
def denoise(x):
    mu, cov = calc_mu_cov(x)
    cov_inv = np.linalg.pinv(cov)
    dist = np.array([distance.mahalanobis(i, mu, cov_inv)**2 for i in x])
    thre =stats.chi2.isf(0.1, 2)
    x_denoise = x[dist < thre]

    return x_denoise


class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    def __init__(self, track_id, bbox, feature, mask, n_init, max_age, reinit_interval):
        self.track_id = track_id
        self.state = TrackState.Tentative
        self.hits = 1
        self.matches = 1
        self.age = 1
        self.no_match_num = 0
        self.bbox = bbox # x, y, w, h
        self.feature = feature
        self.mask = mask
        self.points = []
        self.center = [self.bbox[0] + self.bbox[2] / 2, self.bbox[1] + self.bbox[3] / 2]
        self.n_init = n_init
        self.max_age = max_age
        self.reinit_interval = reinit_interval

    def update_status_every(self):
        self.age += 1

    def update_status_hits(self): # for starting track      
        self.hits += 1
        if self.state == TrackState.Tentative and self.hits >= self.n_init:
            self.state = TrackState.Confirmed

    def update_by_points(self, new_points):
        # Update position (x, y)
        delta = new_points - self.points
        delta_norm = [np.linalg.norm(np.array(i)) for i in delta.tolist()]
        median = statistics.median_low(delta_norm)
        median_ind = delta_norm.index(median)
        self.bbox[0] = self.bbox[0] + delta[median_ind][0]
        self.bbox[1] = self.bbox[1] + delta[median_ind][1]

        # Update scale (w, h)
#        if len(new_points) >= 2:
#            new_points_denoise = denoise(new_points)
#            _, cov_new = calc_mu_cov(new_points_denoise)
#            cov_new += 0.01 * np.ones([2, 2])
#            old_points_denoise = denoise(old_points)
#            _, cov_old = calc_mu_cov(old_points_denoise)
#            cov_old += 0.01 * np.ones([2, 2])
#            self.bbox[2] = self.bbox[2] * cov_new[0][0] / cov_old[0][0]
#            self.bbox[3] = self.bbox[3] * cov_new[1][1] / cov_old[1][1]

        # Update points
        self.points = new_points

        # Update status
        self.update_status_hits()

    def update_by_det(self, det, feature, mask, already_updated):
        self.bbox = det
        self.feature = feature
        self.mask = mask
        self.no_match_num = 0
        if already_updated == 0:
            self.update_status_hits()

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        self.no_match_num += 1

        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.no_match_num >= self.max_age / self.reinit_interval:
            self.state = TrackState.Deleted

    def mark_deleted(self):
        """Mark this track as deleted
        """
        self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
