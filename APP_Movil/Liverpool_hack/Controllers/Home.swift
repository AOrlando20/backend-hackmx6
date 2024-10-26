//
//  ViewController.swift
//  Liverpool_hack
//
//  Created by Alexis Chávez on 25/10/24.
//

import UIKit

class Home: UIViewController, UITabBarDelegate {
    @IBOutlet weak var text1: UILabel!
    @IBOutlet weak var text2: UILabel!
    
    let tabBar: UITabBar = {
        let tabBar = UITabBar()
        let tabItem1 = UITabBarItem(title: "Inicio", image: UIImage(systemName: "house"), tag: 1)
        let tabItem2 = UITabBarItem(title: "Explorar", image: UIImage(systemName: "mail.and.text.magnifyingglass"), tag: 2)
        let tabItem3 = UITabBarItem(title: "Mi cuenta", image: UIImage(systemName: "person.crop.circle"), tag: 3)
        tabBar.items = [tabItem1, tabItem2, tabItem3]
        tabBar.selectedItem = tabItem1
        return tabBar
    }()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupTabBar()
        setupTextFonts()
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
        let labelFontMedium = UIFont.systemFont(ofSize: 20, weight: .medium)
        let labelFontHeavy = UIFont.systemFont(ofSize: 30, weight: .heavy)

        text1.font = labelFontHeavy
        text1.textColor = UIColor.rosa
        text2.font = labelFontMedium
        text2.textColor = UIColor.rosa
        
        let labels = [text1, text2]
        
        for label in labels {
            label?.font = UIFont(name: "Avenir-Heavy", size: 20)
            label?.textColor = UIColor.darkGray
        }
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
}

