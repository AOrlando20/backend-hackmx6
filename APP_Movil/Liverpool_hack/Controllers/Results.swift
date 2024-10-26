//
//  Result.swift
//  Liverpool_hack
//
//  Created by Alexis Chávez on 26/10/24.
//
import UIKit

class Results: UIViewController, UITableViewDelegate, UITableViewDataSource, UITabBarDelegate {

    var similarProducts: [Explore.SimilarProduct] = []
    var products: [Product] = []

    var tableView: UITableView!
    @IBOutlet weak var related: UILabel!
    
    let tabBar: UITabBar = {
        let tabBar = UITabBar()
        let tabItem1 = UITabBarItem(title: "Inicio", image: UIImage(systemName: "house"), tag: 1)
        let tabItem2 = UITabBarItem(title: "Explorar", image: UIImage(systemName: "mail.and.text.magnifyingglass"), tag: 2)
        let tabItem3 = UITabBarItem(title: "Mi cuenta", image: UIImage(systemName: "person.crop.circle"), tag: 3)
        tabBar.items = [tabItem1, tabItem2, tabItem3]
        tabBar.selectedItem = tabItem2
        return tabBar
    }()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupView()
        setupTextFonts()
        setupTabBar()
        setupTableView()
        loadProducts()
    }
    func setupTextFonts() {
        let labels = [related]
        
        for label in labels {
            label?.font = UIFont(name: "Avenir-Heavy", size: 26)
            label?.textColor = UIColor.darkGray
        }
    }
    private func setupView() {
        view.backgroundColor = UIColor.white
        self.modalPresentationStyle = .overFullScreen
    }

    func setupTabBar() {
        if let items = tabBar.items {
            for item in items {
                item.setTitleTextAttributes([NSAttributedString.Key.font: UIFont.systemFont(ofSize: 10, weight: .heavy)], for: .normal)
                item.setTitleTextAttributes([NSAttributedString.Key.font: UIFont.systemFont(ofSize: 12, weight: .medium)], for: .selected)
            }
        }
        tabBar.delegate = self
        tabBar.barTintColor = UIColor(named: "Rosa")
        tabBar.tintColor = .white
        tabBar.unselectedItemTintColor = .white
        
        tabBar.layer.shadowColor = UIColor.black.cgColor
        tabBar.layer.shadowOffset = CGSize(width: 0, height: -2)
        tabBar.layer.shadowRadius = 10
        tabBar.layer.shadowOpacity = 0.3
        tabBar.translatesAutoresizingMaskIntoConstraints = false
        
        view.addSubview(tabBar)
        
        NSLayoutConstraint.activate([
            tabBar.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor),
            tabBar.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor),
            tabBar.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor),
            tabBar.heightAnchor.constraint(equalToConstant: 60)
        ])
    }
    
    private func setupTableView() {
        tableView = UITableView(frame: .zero, style: .plain)
        tableView.delegate = self
        tableView.dataSource = self
        tableView.register(ProductTableViewCell.self, forCellReuseIdentifier: "ProductCell")
        tableView.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(tableView)

        NSLayoutConstraint.activate([

            tableView.topAnchor.constraint(equalTo: related.bottomAnchor, constant: 15),
            tableView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            tableView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            tableView.bottomAnchor.constraint(equalTo: tabBar.topAnchor)
        ])
        
        tableView.backgroundColor = UIColor.systemGroupedBackground
    }


    func printSkus() {
        for product in similarProducts {
            print("SKU: \(product.sku)")
            if let productName = getProductName(by: product.sku) {
                print("Nombre del producto: \(productName)")
            } else {
                print("Producto no encontrado para SKU: \(product.sku)")
            }
        }
    }

    func loadProducts() {
        guard let url = Bundle.main.url(forResource: "output", withExtension: "json") else {
            print("Error: Could not find the JSON file.")
            return
        }

        do {
            let data = try Data(contentsOf: url)
            let decoder = JSONDecoder()
            products = try decoder.decode([Product].self, from: data)
            tableView.reloadData()
        } catch {
            print("Error loading products: \(error)")
        }
    }

    func getProductName(by sku: String) -> String? {
        for product in products {
            if product.sku == sku {
                return product.name
            }
        }
        return nil
    }

    func updateResults(with skus: [Explore.SimilarProduct]) {
        self.similarProducts = skus
    }
    
    // MARK: - UITabBarDelegate
    func tabBar(_ tabBar: UITabBar, didSelect item: UITabBarItem) {
        switch item.tag {
        case 1:
            let storyboard = UIStoryboard(name: "Main", bundle: nil)
            if let transtition2 = storyboard.instantiateViewController(withIdentifier: "Home") as? Home {
                transtition2.modalPresentationStyle = .fullScreen
                self.present(transtition2, animated: false, completion: nil)
            }
        case 2:
            let storyboard = UIStoryboard(name: "Main", bundle: nil)
            if let transtition2 = storyboard.instantiateViewController(withIdentifier: "Explore") as? Explore {
                transtition2.modalPresentationStyle = .fullScreen
                self.present(transtition2, animated: false, completion: nil)
            }
        case 3:
            let storyboard = UIStoryboard(name: "Main", bundle: nil)
            if let transtition2 = storyboard.instantiateViewController(withIdentifier: "Profile") as? Profile {
                transtition2.modalPresentationStyle = .fullScreen
                self.present(transtition2, animated: false, completion: nil)
            }
        default:
            break
        }
    }

    // MARK: - UITableViewDataSource Methods

    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return similarProducts.count
    }

    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "ProductCell", for: indexPath) as! ProductTableViewCell
        let product = similarProducts[indexPath.row]
        if let productName = getProductName(by: product.sku) {
            cell.configure(name: productName)
        } else {
            cell.configure(name: "Producto no encontrado")
        }
        return cell
    }

    // MARK: - UITableViewDelegate Methods

    func tableView(_ tableView: UITableView, heightForRowAt indexPath: IndexPath) -> CGFloat {
        return UITableView.automaticDimension
    }

    func tableView(_ tableView: UITableView, estimatedHeightForRowAt indexPath: IndexPath) -> CGFloat {
        return 44 // Estimación inicial
    }
}

// MARK: - Custom Cell Class
class ProductTableViewCell: UITableViewCell {
    private let nameLabel: UILabel = {
        let label = UILabel()
        label.translatesAutoresizingMaskIntoConstraints = false
        label.textAlignment = .left
        label.font = UIFont.boldSystemFont(ofSize: 18)
        label.textColor = UIColor.darkGray
        label.numberOfLines = 0
        return label
    }()
    
    override init(style: UITableViewCell.CellStyle, reuseIdentifier: String?) {
        super.init(style: style, reuseIdentifier: reuseIdentifier)
        contentView.addSubview(nameLabel)
        setupConstraints()
        contentView.layer.borderColor = UIColor.rosa.cgColor
        contentView.layer.borderWidth = 1
        contentView.layer.cornerRadius = 0
        contentView.backgroundColor = UIColor.white
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    private func setupConstraints() {
        NSLayoutConstraint.activate([
            nameLabel.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 8),
            nameLabel.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 16),
            nameLabel.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -16),
            nameLabel.bottomAnchor.constraint(equalTo: contentView.bottomAnchor, constant: -8)
        ])
    }

    func configure(name: String) {
        nameLabel.text = name
    }
}
