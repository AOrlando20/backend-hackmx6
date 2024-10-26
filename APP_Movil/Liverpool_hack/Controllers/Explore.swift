//
//  ViewController.swift
//  Liverpool_hack
//
//  Created by Alexis Chávez on 25/10/24.
//

import UIKit

class Explore: UIViewController, UITabBarDelegate, UICollectionViewDataSource, UICollectionViewDelegateFlowLayout, UIImagePickerControllerDelegate, UINavigationControllerDelegate   {
    @IBOutlet weak var searchBar: UISearchBar!
    @IBOutlet weak var text1: UILabel!
    
    struct SimilarProduct {
        let sku: String
        let similarity: Double
    }
    var imageToSearch: UIImage?
    var someImage = UIImage()
    let tabBar: UITabBar = {
        let tabBar = UITabBar()
        let tabItem1 = UITabBarItem(title: "Inicio", image: UIImage(systemName: "house"), tag: 1)
        let tabItem2 = UITabBarItem(title: "Explorar", image: UIImage(systemName: "mail.and.text.magnifyingglass"), tag: 2)
        let tabItem3 = UITabBarItem(title: "Mi cuenta", image: UIImage(systemName: "person.crop.circle"), tag: 3)
        tabBar.items = [tabItem1, tabItem2, tabItem3]
        tabBar.selectedItem = tabItem2
        return tabBar
    }()
    private let cameraButton: UIButton = {
        let button = UIButton()
        button.setImage(UIImage(systemName: "camera.badge.clock"), for: .normal)
        button.tintColor = .rosa
        button.addTarget(self, action: #selector(cameraButtonTapped), for: .touchUpInside)
        return button
    }()
    
    private let galleryButton: UIButton = {
        let button = UIButton()
        button.setImage(UIImage(systemName: "photo.badge.plus"), for: .normal)
        button.tintColor = .rosa
        button.addTarget(self, action: #selector(galleryButtonTapped), for: .touchUpInside)
        return button
    }()
    
    var collectionView: UICollectionView!
    var products: [Product] = []
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupTabBar()
        setupCollectionView()
        loadProducts()
        setupTextFonts()
        setupButtons()
        if let image = imageToSearch {
            searchForSimilarProducts(image: image)
        } else {
            print("No hay imagen para buscar.")
        }
    }
    func setupButtons() {
        let buttonStack = UIStackView(arrangedSubviews: [cameraButton, galleryButton])
        buttonStack.axis = .horizontal
        buttonStack.spacing = 10
        buttonStack.translatesAutoresizingMaskIntoConstraints = false
        
        view.addSubview(buttonStack)
        
        NSLayoutConstraint.activate([
            buttonStack.centerYAnchor.constraint(equalTo: text1.centerYAnchor),
            buttonStack.leadingAnchor.constraint(equalTo: text1.trailingAnchor, constant: 20),
            buttonStack.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -10)
        ])
    }
    
    @objc func cameraButtonTapped() {
        if UIImagePickerController.isSourceTypeAvailable(.camera) {
            let imagePicker = UIImagePickerController()
            imagePicker.delegate = self
            imagePicker.sourceType = .camera
            imagePicker.allowsEditing = true
            present(imagePicker, animated: true, completion: nil)
        } else {
            print("Camera is not available")
        }
    }
    
    @objc func galleryButtonTapped() {
        let imagePicker = UIImagePickerController()
        imagePicker.delegate = self
        imagePicker.sourceType = .photoLibrary
        imagePicker.allowsEditing = true
        present(imagePicker, animated: true, completion: nil)
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
    func setupTextFonts() {
        let labelFontHeavy = UIFont.systemFont(ofSize: 18, weight: .medium)
        
        text1.font = labelFontHeavy
        text1.textColor = UIColor.rosa
        
        let labels = [text1]
        
        for label in labels {
            label?.font = UIFont(name: "Avenir-Heavy", size: 18)
            label?.textColor = UIColor.darkGray
        }
    }
    func setupCollectionView() {
        let layout = UICollectionViewFlowLayout()
        layout.sectionInset = UIEdgeInsets(top: 10, left: 10, bottom: 10, right: 10)
        collectionView = UICollectionView(frame: .zero, collectionViewLayout: layout)
        collectionView.translatesAutoresizingMaskIntoConstraints = false
        collectionView.backgroundColor = .white
        collectionView.dataSource = self
        collectionView.delegate = self
        
        collectionView.register(ExploreCollectionViewCell.self, forCellWithReuseIdentifier: "ExploreCell")
        
        view.addSubview(collectionView)
        
        NSLayoutConstraint.activate([
            collectionView.topAnchor.constraint(equalTo: text1.bottomAnchor, constant: 10),
            collectionView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            collectionView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            collectionView.bottomAnchor.constraint(equalTo: tabBar.topAnchor)
        ])
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
            collectionView.reloadData()
        } catch {
            print("Error loading products: \(error)")
        }
    }
    
    // MARK: - UICollectionViewDataSource
    func collectionView(_ collectionView: UICollectionView, numberOfItemsInSection section: Int) -> Int {
        return products.count
    }
    
    func collectionView(_ collectionView: UICollectionView, cellForItemAt indexPath: IndexPath) -> UICollectionViewCell {
        let cell = collectionView.dequeueReusableCell(withReuseIdentifier: "ExploreCell", for: indexPath) as! ExploreCollectionViewCell
        let product = products[indexPath.item]
        cell.configure(with: product.imagesArray, title: product.name)
        return cell
    }
    
    // MARK: - UICollectionViewDelegateFlowLayout
    func collectionView(_ collectionView: UICollectionView, layout collectionViewLayout: UICollectionViewLayout, sizeForItemAt indexPath: IndexPath) -> CGSize {
        let width = (view.frame.width - 40) / 2
        return CGSize(width: width, height: 200)
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
    // MARK: - UIImagePickerControllerDelegate
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        let selectedButton = picker.sourceType == .camera ? cameraButton : galleryButton
        let otherButton = picker.sourceType == .camera ? galleryButton : cameraButton
        
        if let editedImage = info[.editedImage] as? UIImage {
            print("Image edited and selected")
            updateButtonAppearance(button: selectedButton, color: .systemGreen, icon: "checkmark.circle.fill")
            resetButtonAppearance(button: otherButton)
            someImage = editedImage
        } else if let originalImage = info[.originalImage] as? UIImage {
            print("Original image selected")
            updateButtonAppearance(button: selectedButton, color: .systemGreen, icon: "checkmark.circle.fill")
            resetButtonAppearance(button: otherButton)
            someImage = originalImage
        }
        
        dismiss(animated: true) {
            self.searchSimilarProducts(with: self.someImage) { skus, error in
                DispatchQueue.main.async {
                    if let error = error {
                        print("Error:", error.localizedDescription)
                    } else if let skus = skus {
                        print("SKUs de productos similares:", skus)
                        // Navegar a la vista de resultados
                        self.navigateToResults(with: skus)
                    } else {
                        print("Sin datos de SKUs")
                    }
                }
            }
        }
    }

    private func navigateToResults(with skus: [SimilarProduct]) {
        let storyboard = UIStoryboard(name: "Main", bundle: nil)
        if let resultsVC = storyboard.instantiateViewController(withIdentifier: "Results") as? Results {
            resultsVC.updateResults(with: skus)
            resultsVC.modalPresentationStyle = .fullScreen
            self.present(resultsVC, animated: true, completion: nil)
        }
    }

    
    private func updateButtonAppearance(button: UIButton, color: UIColor, icon: String) {
        button.tintColor = color
        button.setImage(UIImage(systemName: icon), for: .normal)
    }
    
    private func resetButtonAppearance(button: UIButton) {
        let originalColor: UIColor = .systemBlue
        let originalIcon: String = (button == cameraButton) ? "camera" : "photo.on.rectangle"
        button.tintColor = originalColor
        button.setImage(UIImage(systemName: originalIcon), for: .normal)
    }
    
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        dismiss(animated: true, completion: nil)
    }
    
    
    func searchForSimilarProducts(image: UIImage) {
        searchSimilarProducts(with: image) { (result, error) in
            if let error = error {
                print("Error al buscar productos similares: \(error.localizedDescription)")
                
                return
            }
            
            guard let products = result else {
                print("No se recibieron productos.")
                return
            }
            
            for product in products {
                print("SKU: \(product.sku), Similarity: \(product.similarity)")
                
            }
        }
    }
    
    func searchSimilarProducts(with image: UIImage, completion: @escaping ([SimilarProduct]?, Error?) -> Void) {
        guard let url = URL(string: "http://34.28.154.130:8000/data_search") else {
            print("URL inválida")
            return
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        guard let imageData = image.jpegData(compressionQuality: 0.8) else {
            print("Error al convertir imagen a datos")
            return
        }
        
        var body = Data()
        let fileName = "image.jpg"
        let fieldName = "file"
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"\(fieldName)\"; filename=\"\(fileName)\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(imageData)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        
        request.httpBody = body
        
        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                DispatchQueue.main.async {
                    completion(nil, error)
                }
                return
            }
            
            guard let data = data else {
                let noDataError = NSError(domain: "dataError", code: -1, userInfo: [NSLocalizedDescriptionKey: "No data received"])
                DispatchQueue.main.async {
                    completion(nil, noDataError)
                }
                return
            }
            
            do {
                
                if let jsonArray = try JSONSerialization.jsonObject(with: data, options: []) as? [[String: Any]] {
                    var products: [SimilarProduct] = []
                    
                    for item in jsonArray {
                        if let sku = item["sku"] as? String, let similarity = item["similarity"] as? Double {
                            products.append(SimilarProduct(sku: sku, similarity: similarity))
                        }
                    }
                    
                    DispatchQueue.main.async {
                        completion(products, nil)
                    }
                } else {
                    let invalidJsonError = NSError(domain: "jsonError", code: -1, userInfo: [NSLocalizedDescriptionKey: "Invalid JSON structure"])
                    DispatchQueue.main.async {
                        completion(nil, invalidJsonError)
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    completion(nil, error)
                }
            }
        }
        
        task.resume()
    }
    
    
    
}

