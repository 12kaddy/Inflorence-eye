# Inflorence-eye

# import the libraries 
    import imageio
    import matplotlib.animation as ani
    import matplotlib.cm as cmx
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    import numpy as np

    from matplotlib.patches import Ellipse
    from PIL import Image
    from sklearn import datasets
    from sklearn.cluster import KMeans
# Iris dataset

    iris = datasets.load_iris()
    eye = iris.data
    eye
#  implementing the Gaussian density function
    def gaussian(eye, mu, cov):
        n = eye.shape[1]
        diff = (eye - mu).T
        return np.diagonal(1 / ((2 * np.pi) ** (n / 2) * np.linalg.det(cov) ** 0.5) * np.exp(-0.5 * np.dot(np.dot(diff.T, np.linalg.inv(cov)), diff))).reshape(-1, 1)
    
    x0 = np.array([[0.05, 1.413, 0.212], [0.85, -0.3, 1.11], [11.1, 0.4, 1.5], [0.27, 0.12, 1.44], [88, 12.33, 1.44]])
    mu = np.mean(x0, axis=0)
    cov = np.dot((x0 - mu).T, x0 - mu) / (x0.shape[0] - 1)

    y = gaussian(x0, mu=mu, cov=cov)
    y
# initialization step of the GMM

    def initialize_clusters(eye, n_clusters):
        clusters = []
        idx = np.arange(eye.shape[0])
    
    
    
    kmeans = KMeans(n_clusters).fit(eye)
    mu_k = kmeans.cluster_centers_
    
    for i in range(n_clusters):
        clusters.append({
            'pi_k': 1.0 / n_clusters,
            'mu_k': mu_k[i],
            'cov_k': np.identity(eye.shape[1], dtype=np.float64)
        })
        
    return clusters
    
# Expectation step    
    
    def expectation_step(eye, clusters):
    totals = np.zeros((eye.shape[0], 1), dtype=np.float64)
    
    for cluster in clusters:
        pi_k = cluster['pi_k']
        mu_k = cluster['mu_k']
        cov_k = cluster['cov_k']
        
        gamma_nk = (pi_k * gaussian(eye, mu_k, cov_k)).astype(np.float64)
        
        for i in range(eye.shape[0]):
            totals[i] += gamma_nk[i]
        
        cluster['gamma_nk'] = gamma_nk
        cluster['totals'] = totals
        
    
    for cluster in clusters:
        cluster['gamma_nk'] /= cluster['totals']
      
# Maximization step      
        def maximization_step(eye, clusters):
    N = float(eye.shape[0])
  
    for cluster in clusters:
        gamma_nk = cluster['gamma_nk']
        cov_k = np.zeros((eye.shape[1], eye.shape[1]))
        
        N_k = np.sum(gamma_nk, axis=0)
        
        pi_k = N_k / N
        mu_k = np.sum(gamma_nk * eye, axis=0) / N_k
        
        for j in range(eye.shape[0]):
            diff = (eye[j] - mu_k).reshape(-1, 1)
            cov_k += gamma_nk[j] * np.dot(diff, diff.T)
            
        cov_k /= N_k
        
        cluster['pi_k'] = pi_k
        cluster['mu_k'] = mu_k
        cluster['cov_k'] = cov_k
        
        
        def get_likelihood(eye, clusters):
    sample_likelihoods = np.log(np.array([cluster['totals'] for cluster in clusters]))
    return np.sum(sample_likelihoods), sample_likelihoods
    
    
    def train_gmm(eye, n_clusters, n_epochs):
    clusters = initialize_clusters(eye, n_clusters)
    likelihoods = np.zeros((n_epochs, ))
    scores = np.zeros((eye.shape[0], n_clusters))
    history = []

    for i in range(n_epochs):
        clusters_snapshot = []
        
        for cluster in clusters:
            clusters_snapshot.append({
                'mu_k': cluster['mu_k'].copy(),
                'cov_k': cluster['cov_k'].copy()
            })
            
        history.append(clusters_snapshot)
      
        expectation_step(eye, clusters)
        maximization_step(eye, clusters)

        likelihood, sample_likelihoods = get_likelihood(eye, clusters)
        likelihoods[i] = likelihood

        print('Epoch: ', i + 1, 'Likelihood: ', likelihood)
        
    for i, cluster in enumerate(clusters):
        scores[:, i] = np.log(cluster['gamma_nk']).reshape(-1)
        
    return clusters, likelihoods, scores, sample_likelihoods, history
    
    
    n_clusters = 3
    n_epochs = 50

    clusters, likelihoods, scores, sample_likelihoods, history = train_gmm(eye, n_clusters, n_epochs)

    plt.figure(figsize=(10, 10))
    plt.title('Log-Likelihood')
    plt.plot(np.arange(1, n_epochs + 1), likelihoods)
    plt.show()

    def create_cluster_animation(eye, history, scores):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    colorset = ['blue', 'red', 'black']
    images = []
    
    for j, clusters in enumerate(history):
      
        idx = 0
      
        if j % 3 != 0:
            continue
        
        plt.cla()
        
        for cluster in clusters:
            mu = cluster['mu_k']
            cov = cluster['cov_k']

            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            order = eigenvalues.argsort()[::-1]
            eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
            vx, vy = eigenvectors[:,0][0], eigenvectors[:,0][1]
            theta = np.arctan2(vy, vx)

            color = colors.to_rgba(colorset[idx])

            for cov_factor in range(1, 4):
                ell = Ellipse(xy=mu, width=np.sqrt(eigenvalues[0]) * cov_factor * 2, height=np.sqrt(eigenvalues[1]) * cov_factor * 2, angle=np.degrees(theta), linewidth=2)
                ell.set_facecolor((color[0], color[1], color[2], 1.0 / (cov_factor * 4.5)))
                ax.add_artist(ell)

            ax.scatter(cluster['mu_k'][0], cluster['mu_k'][1], c=colorset[idx], s=1000, marker='+')
            idx += 1

        for i in range(eye.shape[0]):
            ax.scatter(eye[i, 0], eye[i, 1], c=colorset[np.argmax(scores[i])], marker='o')
        
        fig.canvas.draw()
        
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        images.append(image)
    
    kwargs_write = {'fps':1.0, 'quantizer':'nq'}
    imageio.mimsave('./gmm.gif', images, fps=1)
    plt.show(Image.open('gmm.gif').convert('RGB'))
    
    
    create_cluster_animation(eye, history, scores)




    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(n_components=n_clusters).fit(eye)
    gmm_scores = gmm.score_samples(eye)

    print('Means by sklearn:\n', gmm.means_)
    print('Means by our implementation:\n', np.array([cluster['mu_k'].tolist() for cluster in clusters]))
    print('Scores by sklearn:\n', gmm_scores[0:20])
    print('Scores by our implementation:\n', sample_likelihoods.reshape(-1)[0:20])
