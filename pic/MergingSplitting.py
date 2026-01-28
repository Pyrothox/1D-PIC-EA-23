import numpy as np
from pic.functions import get_particle_indexes_in_cells


class DPMSA:
    def __init__(self, plasma, cluster_size_min=4, merge_interval=10000):
        self.plasma = plasma
        self.N_min = cluster_size_min
        self.merge_interval = merge_interval

    # ---------- metric: Minkowski l=1 (1D–3V) ----------
    def distance(self, x1, x2, p1, p2):
        return abs(x1 - x2) + np.sum(np.abs(p1 - p2))

    # ---------- Step 1: initialize K0 mass centers in a cell ----------
    def init_centers_in_cell(self, x, p, w, indices):
        indices = np.array(indices, dtype=int)
        Nc = indices.size
        K0 = max(1, int(np.sqrt(Nc / 2.0)))

        centers = [int(np.random.choice(indices))]
        while len(centers) < K0:
            D_min = np.empty(Nc)
            for j_idx, i in enumerate(indices):
                dists = [self.distance(x[i], x[c], p[i], p[c]) for c in centers]
                D_min[j_idx] = np.min(dists)
            D_sq = D_min ** 2
            S = np.sum(D_sq)
            probs = np.ones(Nc) / Nc if S == 0.0 else D_sq / S
            next_idx = np.random.choice(np.arange(Nc), p=probs)
            cand = int(indices[next_idx])
            if cand not in centers:
                centers.append(cand)
            else:
                break
        return np.array(centers, dtype=int), K0

    # ---------- one pass of Step 2: sort + enforce N_min + recompute centers ----------
    def sort_particles_in_cell(self, x, p, w, indices, centers):
        indices = np.array(indices, dtype=int)
        centers = np.array(centers, dtype=int)

        # assign particles to nearest center
        clusters = {int(c): [] for c in centers}
        for i in indices:
            dists = [self.distance(x[i], x[c], p[i], p[c]) for c in centers]
            c_near = int(centers[int(np.argmin(dists))])
            clusters[c_near].append(int(i))

        # enforce minimum cluster size
        for c in list(clusters.keys()):
            if len(clusters[c]) < self.N_min:
                # move this cluster to nearest other center
                # only consider other centers that still exist in clusters
                other_centers = [int(oc) for oc in centers if int(oc) != int(c) and int(oc) in clusters]
                if not other_centers:
                    continue
                dmin = np.inf
                best = None
                for oc in other_centers:
                    oc = int(oc)
                    d = self.distance(x[int(c)], x[int(oc)], p[int(c)], p[int(oc)])
                    if d < dmin:
                        dmin = d
                        best = int(oc)
                if best is not None and best in clusters:
                    clusters[best].extend(clusters[c])
                    del clusters[c]

        centers_after = np.array(list(clusters.keys()), dtype=int)

        # recompute mass centers by weighted average
        new_centers = []
        new_clusters = {}
        for c in centers_after:
            group = np.array(clusters[c], dtype=int)
            W = np.sum(w[group])
            if W == 0.0 or group.size == 0:
                continue
            x_c = np.sum(x[group] * w[group]) / W
            p_c = np.sum(p[group] * w[group][:, None], axis=0) / W
            d_to_center = [self.distance(x[i], x_c, p[i], p_c) for i in group]
            i_best = group[int(np.argmin(d_to_center))]
            new_centers.append(i_best)
            new_clusters[i_best] = list(group)

        return np.array(new_centers, dtype=int), new_clusters

    # ---------- Step 3: splitting ----------
    def split_clusters(self, x, p, w, centers, clusters):
        sig = {}
        for c in centers:
            idx = np.array(clusters[c], dtype=int)
            if idx.size == 0:
                sig[c] = 0.0
                continue
            W = np.sum(w[idx])
            if W == 0.0:
                sig[c] = 0.0
                continue
            P = np.sum(p[idx] * w[idx][:, None], axis=0) / W
            diff = p[idx] - P
            var = np.sum(w[idx][:, None] * diff ** 2, axis=0) / W
            sig[c] = np.sqrt(np.max(var))
        if not sig:
            return centers, clusters

        alpha = np.mean(list(sig.values()))

        new_clusters = {c: list(clusters[c]) for c in centers}
        new_centers = list(centers)

        for c in list(centers):
            idx = np.array(new_clusters.get(c, []), dtype=int)
            if idx.size <= 2 * self.N_min or sig[c] < alpha:
                continue

            W = np.sum(w[idx])
            P = np.sum(p[idx] * w[idx][:, None], axis=0) / W
            diff = p[idx] - P
            var = np.sum(w[idx][:, None] * diff ** 2, axis=0) / W
            j_max = int(np.argmax(var))
            proj = p[idx, j_max]
            thr = np.median(proj)
            left = idx[proj <= thr]
            right = idx[proj > thr]

            if left.size < self.N_min or right.size < self.N_min:
                continue

            def make_center(sub):
                Wsub = np.sum(w[sub])
                x_c = np.sum(x[sub] * w[sub]) / Wsub
                p_c = np.sum(p[sub] * w[sub][:, None], axis=0) / Wsub
                dlist = [self.distance(x[i], x_c, p[i], p_c) for i in sub]
                return sub[int(np.argmin(dlist))], list(sub)

            cL, gL = make_center(left)
            cR, gR = make_center(right)
            new_centers.append(cL)
            new_centers.append(cR)
            new_clusters[cL] = gL
            new_clusters[cR] = gR
            del new_clusters[c]

        return np.array(new_centers, dtype=int), new_clusters

    # ---------- Step 4: merging ----------
    def merge_clusters(self, x, p, centers, clusters):
        if not clusters:
            return centers, clusters

        all_idx = np.concatenate([np.array(v, int) for v in clusters.values()])
        d_list = []
        for i in range(all_idx.size):
            for j in range(i + 1, all_idx.size):
                a, b = all_idx[i], all_idx[j]
                d_list.append(self.distance(x[a], x[b], p[a], p[b]))
        d_min = np.mean(d_list) if d_list else 0.0

        centers_list = list(centers)
        merged = {c: list(clusters[c]) for c in centers_list}
        used = set()

        for i in range(len(centers_list)):
            ci = centers_list[i]
            if ci in used:
                continue
            for j in range(i + 1, len(centers_list)):
                cj = centers_list[j]
                if cj in used:
                    continue
                Dij = self.distance(x[ci], x[cj], p[ci], p[cj])
                if Dij <= d_min:
                    merged[ci].extend(merged[cj])
                    used.add(cj)
                    del merged[cj]

        new_centers = np.array(list(merged.keys()), dtype=int)
        return new_centers, merged

    # ---------- cell-level DPMSA loop (steps 1–5) ----------
    def process_cell(self, x, p, w, indices):
        indices = np.array(indices, dtype=int)
        Nc = indices.size
        if Nc <= self.N_min:
            return x[indices], p[indices], w[indices]

        centers, K0 = self.init_centers_in_cell(x, p, w, indices)
        centers, clusters = self.sort_particles_in_cell(x, p, w, indices, centers)

        while True:
            K = len(centers)
            if K <= K0 / 2:
                centers, clusters = self.split_clusters(x, p, w, centers, clusters)
                centers, clusters = self.sort_particles_in_cell(x, p, w, indices, centers)
            elif K >= 2 * K0:
                centers, clusters = self.merge_clusters(x, p, centers, clusters)
                centers, clusters = self.sort_particles_in_cell(x, p, w, indices, centers)
            else:
                break

        x_new, p_new, w_new = [], [], []
        for c, group in clusters.items():
            idx = np.array(group, dtype=int)
            if idx.size == 1:
                i = idx[0]
                x_new.append(x[i])
                p_new.append(p[i])
                w_new.append(w[i])
            else:
                W = np.sum(w[idx])
                x_c = np.sum(x[idx] * w[idx]) / W
                p_c = np.sum(p[idx] * w[idx][:, None], axis=0) / W
                x_new.append(x_c)
                p_new.append(p_c)
                w_new.append(W)

        return np.array(x_new), np.array(p_new), np.array(w_new)

    # ---------- public entry ----------
    def execute(self, nt):
        if nt - 10 % self.merge_interval != 0:
            return
        print(f"DPMSA executing at nt={nt}")
        tabx = self.plasma.x_j
        dx = self.plasma.dx
        Ncells = self.plasma.N_cells

        for specie in self.plasma.species.values():
            Npart = specie.Npart
            if Npart <= self.N_min:
                continue
            x = specie.x[:Npart]
            p = specie.V[:Npart, :]
            w = specie.w[:Npart]

            cells_indexes = get_particle_indexes_in_cells(
                Npart, x, tabx, None, dx
            )

            x_new_all, p_new_all, w_new_all = [], [], []
            for icell in range(Ncells):
                print(f"DPMSA processing cell {icell+1}/{Ncells}, nt={nt}")
                idx = cells_indexes[icell]
                if not idx:
                    continue
                x_c, p_c, w_c = self.process_cell(x, p, w, idx)
                x_new_all.append(x_c)
                p_new_all.append(p_c)
                w_new_all.append(w_c)

            if not x_new_all:
                continue

            x_new_all = np.concatenate(x_new_all)
            p_new_all = np.concatenate(p_new_all)
            w_new_all = np.concatenate(w_new_all)

            n_new = w_new_all.size
            specie.x[:n_new] = x_new_all
            specie.V[:n_new, :] = p_new_all
            specie.w[:n_new] = w_new_all
            specie.Npart = n_new
