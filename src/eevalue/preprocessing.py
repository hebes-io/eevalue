# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def assign_clusters(plants):
    plants = plants.copy()

    if not "Nunits" in plants:
        plants["Nunits"] = 1

    X = plants[
        [
            "MinUpTime",
            "MinDownTime",
            "RampUpRate",
            "RampDownRate",
            "StartUpCost",
            "PartLoadMin",
            "StartUpTime",
        ]
    ]

    scores = []
    for n_clusters in range(2, len(X)):
        clusterer = KMeans(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        scores.append(silhouette_avg)
        if silhouette_avg < 0.6:
            break

    n_clusters = np.argmax(np.array(scores)) + 2
    cluster_labels = KMeans(n_clusters=n_clusters).fit_predict(X)
    plants["Cluster"] = cluster_labels

    cl = n_clusters
    for cluster in np.unique(cluster_labels):
        subset = plants[plants["Cluster"] == cluster]
        combinations = set(zip(subset["Technology"], subset["Fuel"]))
        if len(combinations) > 1:
            for combination in list(combinations)[1:]:
                tech, fuel = combination
                plants.loc[
                    subset[
                        (subset["Technology"] == tech) & (subset["Fuel"] == fuel)
                    ].index,
                    "Cluster",
                ] = cl
                cl += 1
    return plants


def group_plants(plants):
    clustered_plants = plants.groupby("Cluster")["Nunits"].sum().to_frame("Nunits")
    clustered_plants["Fuel"] = (
        plants.groupby("Cluster")["Fuel"].unique().map(lambda x: x[0])
    )
    clustered_plants["Technology"] = (
        plants.groupby("Cluster")["Technology"].unique().map(lambda x: x[0])
    )
    clustered_plants["Units"] = (
        plants.groupby("Cluster")["Unit"].unique().map(lambda x: x.tolist())
    )

    clustered_plants["TotalCapacity"] = plants.groupby("Cluster")["PowerCapacity"].sum()
    clustered_plants["PowerCapacity"] = (
        clustered_plants["TotalCapacity"] / clustered_plants["Nunits"]
    )

    clustered_plants["PowerMinStable"] = plants.groupby("Cluster").apply(
        lambda x: (x["PartLoadMin"] * x["PowerCapacity"]).min()
    )

    clustered_plants["RampStartUpRate"] = clustered_plants[
        "PowerCapacity"
    ] / plants.groupby("Cluster").apply(
        lambda x: (x["PowerCapacity"] * x["StartUpTime"]).sum()
        / x["PowerCapacity"].sum()
    )

    for key in [
        "Efficiency",
        "MinUpTime",
        "MinDownTime",
        "RampUpRate",
        "RampDownRate",
        "CO2Intensity",
    ]:
        clustered_plants[key] = plants.groupby("Cluster").apply(
            lambda x: (x["PowerCapacity"] * x[key]).sum() / x["PowerCapacity"].sum()
        )

    for key in ["RampUpRate", "RampDownRate"]:
        clustered_plants[key] = (
            60 * clustered_plants[key] * clustered_plants["PowerCapacity"]
        )

    ramping_cost = plants.groupby("Cluster").apply(
        lambda x: (x["PowerCapacity"] * x["RampingCost"]).sum()
        / x["PowerCapacity"].sum()
    )

    clustered_plants["CostRampUp"] = ramping_cost + (
        plants.groupby("Cluster").apply(
            lambda x: x["StartUpCost"].sum() / x["PowerCapacity"].sum()
        )
    )

    if "ShutDownCost" in plants.columns:
        clustered_plants["CostRampDown"] = ramping_cost + (
            plants.groupby("Cluster").apply(
                lambda x: x["ShutDownCost"].sum() / x["PowerCapacity"].sum()
            )
        )

    clustered_plants["Cluster"] = clustered_plants[["Technology", "Fuel"]].agg(
        "_".join, axis=1
    )
    return clustered_plants
