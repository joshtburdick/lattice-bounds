#!/usr/bin/env python3
# Plots graphs, after zeroing out some hypercliques.
# FIXME
# - clarify where arrows are pointing

import colorsys
import itertools
import pdb
import random
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import scipy.interpolate
import scipy.stats

# set seed
np.random.seed(12345)

# for consistency, these are set for a bunch of figures
n = 6
cliques = [frozenset(s) for s in itertools.combinations(range(n), 3)]
# sort these lexicographically
cliques.sort()


def random_hypergraph():
    """Picks a random hypergraph.

    Each hyperedge is included with probability 1/2.
    """
    return [e for e in cliques if np.random.choice([True, False])]


def color_with_hue(h):
    """Makes a color with some hue."""
    return colorsys.hsv_to_rgb(h, 0.5, 0.5)


def scale_lightness(rgb, scale_l):
    """Scales the lightness of a color.

    From
    https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
    """
    # convert RGB to HLS
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as RGB
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)


class CliqueFigure:
    """Plots cliques in a figure.

    This assumes that a given hyperedge is the same color
    in each plot.
    """

    def __init__(self, axs, n, colors, vertex_0_theta):
        """
        axs: context on which to plot
        n: number of vertices in each graph
        colors: hash from sets (defined as sorted lists of k numbers)
            to color of each set
        vertex_0_theta: angle at which to place vertex 0
        FIXME
        - include radius here?
        - add method to show which edge was zeroed out?
        """
        self.axs = axs
        self.n = n
        self.colors = colors
        self.alpha = 0.5
        # set the vertices
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False) + vertex_0_theta
        self.vertex = np.stack([np.cos(theta), np.sin(theta)])

    def get_vertices(self, radius, center):
        """Gets the vertices for a set of cliques at some location."""
        return radius * self.vertex + np.array([center]).T

    def plot_cliques(self, radius, center, cliques):
        """Plots some cliques.

        radius: the radius for the cliques
        center: the center, as a 2-element tuple, or np.array of shape (2,)
        cliques: the cliques, as a list of k-element tuples of ints
        Side effects: plots the cliques
        """
        # if there are no cliques, plot an "empty set" sign
        if not cliques:
            self.axs.text(
                np.array(center)[0],
                np.array(center)[1],
                "$\\emptyset$",
                fontsize=10,
                ha="center",
                va="center",
            )
            return
        for s in cliques:
            v = radius * self.vertex[:, list(s)] + np.array([center]).T
            # pdb.set_trace()
            self.axs.fill(
                v[0, :],
                v[1, :],
                edgecolor=self.colors[s],
                facecolor=scale_lightness(self.colors[s], 2),
                lw=1,
                alpha=self.alpha,
                joinstyle="round",
            )


def zero_out_edges(cliques):
    """
    Finds the cliques remaining, after zeroing out an edge.

    """
    # get the vertices which are relevant
    vertices = set(list(itertools.chain(*cliques)))
    # the edges, which we could zero out
    edges = itertools.combinations(vertices, 2)

    def cliques_remaining(edge):
        # note that < is comparing frozensets
        return frozenset([c for c in cliques if not frozenset(edge) < c])

    # note that this removes self-loops
    return list(set([cliques_remaining(e) for e in edges]) - set([cliques]))


def zero_vertex(v, s):
    """Returns a set, with one vertex zeroed out."""
    return [h for h in s if v not in h]


def all_vertices(x):
    """Gets all the vertices in a set of edges."""
    # FIXME replace this with something simpler?
    r = []
    for e in x:
        for v in e:
            r.append(v)
    return frozenset(r)


def num_vertices_zeroed(a, b):
    """Gets number of vertices zeroed to convert b to a.

    a, b: sets of cliques (as lists of frozensets of ints)
    Returns: if some set v of vertices in b can be zeroed
        to result in a, then v; otherwise returns None.
    """
    a = frozenset(a)
    b = frozenset(b)
    # first, make sure that a is a strict subset of b
    if not (a < b):
        return None
    # get all vertices in each
    a_vertices = all_vertices(a)
    b_vertices = all_vertices(b)
    # get the vertices which _might_ be zeroable
    z = b_vertices - a_vertices
    # zero those out of b
    b1 = frozenset([e for e in b if not e.intersection(z)])
    # pdb.set_trace()
    # b is zeroable to a iff this is the same as a
    if a == b1:
        return len(z)
    else:
        return None


# quick test
if False:
    z = has_zeroing_path(
        [frozenset([1, 2, 3])], [frozenset([1, 2, 3]), frozenset([1, 2, 5])]
    )
    print(z)


def plot_bounce():
    """Plots one 'bounce', down and up."""
    # hopefully this will make the figure reproducible
    random.seed(0)
    # edge to be zeroed out
    zeroed_edge = (2, 3)

    # get sets which the edge hits, and misses
    hit_sets = frozenset([c for c in cliques if frozenset(zeroed_edge) <= c])
    missed_sets = frozenset(cliques) - hit_sets

    # construct the sets: first the set after "zeroing"...
    b_t = frozenset(random.sample(sorted(missed_sets), 3))
    # ... then the "before" and "after" sets
    # XXX these are actually now the "extra" cliques, which are
    # zeroed out or added, respectively
    a_t = frozenset(random.sample(sorted(hit_sets), 3))
    a_t_plus_1 = frozenset(random.sample(sorted(hit_sets), 2))

    # set up figure
    plt.figure(figsize=(4, 2.5))
    plt.axis("off")
    # plt.xlim(-0.3,1.3)
    # plt.ylim(-0.3,1.3)

    # utility to set all cliques to one color
    # FIXME maybe simpler if each clique had its own alpha?
    def all_colors(c):
        return {s: c for s in cliques}

    cf = CliqueFigure(plt.gca(), n, all_colors(colorsys.hsv_to_rgb(0, 0, 0.25)), 0)
    radius = 0.4

    # draw cliques
    x1 = np.array([[0.75, 0], [0.25, 1]]).T
    x = np.array(
        [[-1.5, 0, 1.5], [len(a_t) + len(b_t), len(b_t), len(a_t_plus_1) + len(b_t)]]
    )
    x[1, :] *= 0.6

    # first, draw b_t more lightly
    cf.alpha = 0.25
    for i in range(3):
        cf.plot_cliques(radius, x[:, i], b_t)
    # then, add cliques from a_t and a_t_plus_1, darker
    cf.alpha = 0.5
    cf.colors = all_colors(colorsys.hsv_to_rgb(0, 0.75, 0.5))
    cf.plot_cliques(radius, x[:, 0], a_t)
    cf.colors = all_colors(colorsys.hsv_to_rgb(2 / 3, 0.75, 0.5))
    cf.plot_cliques(radius, x[:, 2], a_t_plus_1)

    def draw_arrow(i, j):
        """Draws an arrow from set i to set j."""
        plt.gca().annotate(
            "",
            xy=x[:, j],
            xytext=x[:, i],
            arrowprops=dict(
                arrowstyle="->",
                connectionstyle="angle3,angleA=-20,angleB=70",
                # relpos=(0,0),
                facecolor="#00000080",
                edgecolor="#00000080",
            ),
        )

    draw_arrow(0, 1)
    draw_arrow(1, 2)

    # label them
    plt.text(x[0, 0] - 0.5, x[1, 0], "$A_t$", fontsize=16, ha="right", va="center")
    plt.text(x[0, 1] - 0.5, x[1, 1], "$B_t$", fontsize=16, ha="right", va="center")
    plt.text(x[0, 2] - 0.5, x[1, 2], "$A_{t+1}$", fontsize=16, ha="right", va="center")

    # plot lines showing which edges were zeroed out
    for i in range(3):
        v = cf.get_vertices(radius, x[:, i])
        ends = v[:, zeroed_edge]
        plt.plot(ends[0, :], ends[1, :], "-", c="black", lw=1.5, alpha=0.8)

    plt.savefig("../../bound2/bounce.pdf", bbox_inches="tight")


def plot_zeroing_one_vertex():
    """Plots the effect of zeroing out one vertex.

    That is plots G and Z(G)."""
    g = random.sample(cliques, 8)
    zeroed_vertex = 2
    zeroed_g = [c for c in g if zeroed_vertex not in c]

    # set up figure
    plt.figure(figsize=(4, 4))
    # plt.axis('off')
    plt.xlim(-0.3, 1.3)
    plt.ylim(-0.3, 1.3)
    colors = {
        c: color_with_hue(2 / 3) if c in zeroed_g else colorsys.hsv_to_rgb(0, 0.5, 0.25)
        for c in cliques
    }
    cf = CliqueFigure(plt.gca(), n, colors, 0)
    radius = 0.3

    # draw cliques
    x = np.array([[0.75, 0], [0.25, 1]]).T
    cf.plot_cliques(radius, x[:, 1], g)
    cf.plot_cliques(radius, x[:, 0], zeroed_g)

    plt.gca().annotate(
        "",
        xy=x[:, 0],
        xytext=x[:, 1],
        arrowprops=dict(
            arrowstyle="->",
            connectionstyle="angle3,angleA=20,angleB=70",
            # relpos=(0,0),
            facecolor="#00000040",
            edgecolor="#00000040",
        ),
    )

    # label them
    plt.text(x[0, 1] - 0.4, x[1, 1], "$G$", fontsize=24, ha="right", va="center")
    plt.text(x[0, 0] - 0.4, x[1, 0], "$Z(G)$", fontsize=24, ha="right", va="center")

    # plot lines showing which edges were zeroed out
    v = cf.get_vertices(radius, x[:, 1])
    for i in range(n):
        if i != zeroed_vertex:
            ends = v[:, [zeroed_vertex, i]]
            plt.plot(ends[0, :], ends[1, :], "-", c="red", alpha=0.5)

    # plt.savefig('zeroing_one_vertex.pdf', bbox_inches='tight')
    plt.savefig("zeroing_one_vertex.pdf", bbox_inches="tight")


def plot_Z_relation():
    """Plots the 'zeroing-one-edge' relation."""
    plt.figure(figsize=(7, 6))
    plt.axis("off")
    plt.xlim(-3, 3.5)
    plt.ylim(0, 4.5)
    axs = plt.gca()

    def color1(h):
        return colorsys.hsv_to_rgb(h, 0.5, 0.5)

    colors = {
        frozenset((0, 1, 2)): color1(0 / 4),
        frozenset((0, 1, 3)): color1(1 / 4),
        frozenset((0, 2, 3)): color1(2 / 4),
        frozenset((1, 2, 3)): color1(3 / 4),
    }
    cf = CliqueFigure(axs, 4, colors, 0)
    # lay out coordinates for each set; this will be keyed by set,
    # and its value will be coordinates
    set_location = {}
    all_cliques = list([frozenset(s) for s in itertools.combinations(range(4), 3)])
    for j in range(0, 5):
        print(j)
        # ??? should this be a set rather than a tuple?
        subsets = tuple(itertools.combinations(all_cliques, j))
        for i in range(len(subsets)):
            print(((i, j), subsets[i]))
            # this is mostly centered, but also slightly tilted
            set_location[frozenset(subsets[i])] = (i - len(subsets) / 2 + j / 3, j)
    # plot effects of zeroing out an edge
    for cliques, location in set_location.items():
        cliques_below = zero_out_edges(cliques)
        for c in cliques_below:
            location_1 = set_location[c]
            # p = curved_line(np.array([location, location_1]))
            # plt.plot(p[:,0], p[:,1], c='black', alpha=0.3)

            axs.annotate(
                "",
                xy=location_1,  # [a['x'], a['y']],
                xytext=location,  # [b['x'], b['y']],
                arrowprops=dict(
                    arrowstyle="->",
                    connectionstyle="angle3,angleA=25,angleB=95",
                    facecolor="black",
                    edgecolor="black",
                    alpha=0.25,
                    linewidth=1,
                ),
            )

    # plot the sets
    for cliques, location in set_location.items():
        # pdb.set_trace()
        print("cliques =")
        print(cliques)
        print("location = " + str(location))
        cf.plot_cliques(0.25, location, cliques)
    # cf.plot_sets(0.4, np.array([0,0.1]), [(0,1,2), (0,1,3)])
    plt.savefig("Z.pdf", bbox_inches="tight")


class ZeroingPlot:
    """Plots some sets, with edges showing zeroing."""

    def __init__(self):
        # the sets of cliques to plot
        self.set_list = []

    def add_set(self, s, x, y=None, label=None):
        """Adds one set.

        s: the set (as a list of frozensets of three ints)
        x: the x-coordinate
        y: the y coordinate (if not set, this will be the number of cliques)
        label: the label to use for the set
        Side effects: adds the set, unless it's already been added.
        (If it's already been added, whatever x and label had been specified
        are used; this is to facilitate plotting "sharing" between paths,
        and emphasize that this is a DAG.)
        """
        # fill in y-coordinate
        if not y:
            y = len(s)
        # check if this has been added; if so, do nothing
        if frozenset(s) in [frozenset(s1["s"]) for s1 in self.set_list]:
            return
        # add to the list of sets
        self.set_list.append(dict(s=s, x=x, y=y, label=label))

    def add_chain(self, s, x, vertices):
        """Adds a set of cliques, from zeroing out several vertices.

        s: the original set of cliques
        x: the x-coordinate to "request" (this may be overridden,
            if some sets are already present)
        vertices: the vertices to zero out
        Side effects: adds all of the sets, with successive
            vertices zeroed out
        """
        self.add_set(s, x)
        for v in vertices:
            s = zero_vertex(v, s)
            self.add_set(s, x)

    def plot(self, axs):
        """Draws the cliques and edges."""
        # create object for plotting
        colors = dict(
            zip(
                cliques,
                [
                    color_with_hue(h)
                    for h in np.linspace(0, 1, len(cliques), endpoint=False)
                ],
            )
        )
        cf = CliqueFigure(axs, n, colors, 0)
        # plot the sets of cliques
        # XXX this sometimes almost has collisions, but it seems worth it
        radius = 0.45
        # pdb.set_trace()
        for s in self.set_list:
            cf.plot_cliques(radius, (s["x"], s["y"]), s["s"])
            # possibly add a label
            if s["label"]:
                axs.text(
                    s["x"] - 0.5,
                    s["y"] + 0.1,
                    s["label"],
                    fontsize=10,
                    ha="right",
                    va="center",
                )
        # plot edges, between pairs of sets
        # we try to somewhat reduce the density of edges, but still
        # make sure everything has a connection.
        # a will loop through the cliques, and plot "from" connections
        for b in self.set_list:
            # we pick the min. "num_vertices_zeroed"
            have_drawn_edge = False
            # loop through "number of vertices zeroed"
            for z in range(1, 7):
                # loop through "to" connections
                for a in self.set_list:
                    if num_vertices_zeroed(a["s"], b["s"]) == z:
                        axs.annotate(
                            "",
                            xy=[a["x"], a["y"]],
                            xytext=[b["x"], b["y"]],
                            arrowprops=dict(
                                arrowstyle="->",
                                connectionstyle="angle3,angleA=-30,angleB=70",
                                facecolor="black",
                                edgecolor="black",
                                alpha=0.25,
                                linewidth=1,
                            ),
                        )
                        have_drawn_edge = True
                # if we've drawn an edge, then don't try zeroing more vertices
                if have_drawn_edge:
                    break


class ZeroingBlockDiagram:
    """Plots a 'block' diagram of the effect of zeroing.

    Not yet implemented. Indeed, deprecated...
    """

    def __init__(self):
        self.max_n = 6
        self.k = 3


def plot_Z_with_vertex_zeroing():
    """Plots Z, with vertices zeroed out."""
    plt.figure(figsize=(8, 6))

    # FIXME
    def color1(h):
        return colorsys.hsv_to_rgb(h, 0.5, 0.5)

    fig, axs = plt.subplots()
    # set colors for cliques, in order
    colors = dict(zip(cliques, [color1(h) for h in np.arange(0, 1, 20)]))
    # set up for plotting
    axs.set_xlim(-7, 7)
    axs.set_ylim(-1, 21)
    axs.spines["top"].set_visible(False)
    axs.spines["right"].set_visible(False)
    axs.spines["bottom"].set_visible(False)
    axs.xaxis.set_visible(False)
    axs.yaxis.get_major_locator().set_params(integer=True)
    zp = ZeroingPlot()

    # add sets of cliques...
    # first, add some labeled sets
    zp.add_set(cliques, 0, label="(a) CLIQUE ")

    def f(x, y, z):
        return frozenset([x, y, z])

    zp.add_set([f(0, 1, 2), f(0, 1, 3), f(0, 2, 3), f(1, 2, 3)], 0, label="(b)")
    s1 = [f(0, 1, 2), f(0, 1, 3), f(0, 1, 4), f(0, 1, 5)]
    zp.add_set(s1, 6, label="(c)")
    # add one path of zeroing out all the edges
    zp.add_chain(cliques, 0, [5, 4, 3, 2, 1, 0])

    # add some more random paths
    for x in [-2, 2, -4, 4, -6]:
        g = random_hypergraph()
        zp.add_chain(g, x, [5, 4, 3, 2, 1, 0])
    # add path including s1
    zp.add_chain(s1, 6, [5, 4, 3, 2, 1, 0])

    # plot them
    zp.plot(axs)
    axs.set_ylabel("Number of cliques")

    # add label
    plt.savefig("Z_with_vertex_zeroing.pdf", bbox_inches="tight")


def plot_covering():
    """Plots several coverings of vertices and edges."""
    pass
