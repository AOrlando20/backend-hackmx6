//
//  ViewController.swift
//  Liverpool_hack
//
//  Created by Alexis Chávez on 25/10/24.
//

import UIKit

class Profile: UIViewController, UITabBarDelegate {
    
    @IBOutlet weak var rect1: UIImageView!
    @IBOutlet weak var rect2: UIImageView!
    @IBOutlet weak var rect3: UIImageView!
    @IBOutlet weak var rect4: UIImageView!
    @IBOutlet weak var rect5: UIImageView!
    @IBOutlet weak var rect6: UIImageView!
    @IBOutlet weak var rect7: UIImageView!
    @IBOutlet weak var rect8: UIImageView!
    @IBOutlet weak var image: UIImageView!
    @IBOutlet weak var text1: UILabel!
    @IBOutlet weak var text2: UILabel!
    @IBOutlet weak var text3: UILabel!
    @IBOutlet weak var text4: UILabel!
    @IBOutlet weak var text5: UILabel!
    @IBOutlet weak var text6: UILabel!
    @IBOutlet weak var text7: UILabel!
    @IBOutlet weak var text8: UILabel!
    
    let tabBar: UITabBar = {
        let tabBar = UITabBar()
        let tabItem1 = UITabBarItem(title: "Inicio", image: UIImage(systemName: "house"), tag: 1)
        let tabItem2 = UITabBarItem(title: "Explorar", image: UIImage(systemName: "mail.and.text.magnifyingglass"), tag: 2)
        let tabItem3 = UITabBarItem(title: "Mi cuenta", image: UIImage(systemName: "person.crop.circle"), tag: 3)
        tabBar.items = [tabItem1, tabItem2, tabItem3]
        tabBar.selectedItem = tabItem3
        return tabBar
    }()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupTabBar()
        setupRectangles()
        setupCircularImage()
        setupTextFonts()
    }
    func setupTextFonts() {
        let labels = [text1, text2, text3, text4, text5, text6, text7, text8]
        
        for label in labels {
            label?.font = UIFont(name: "Avenir-Heavy", size: 16)
            label?.textColor = UIColor.darkGray
        }
    }
    func setupCircularImage() {
        image.layer.cornerRadius = image.frame.size.width / 2
        image.layer.masksToBounds = true
    }
    func setupRectangles() {
        let rectangles = [rect1, rect2, rect3, rect4, rect5, rect6, rect7, rect8]
        
        for rect in rectangles {
            rect?.backgroundColor = UIColor(white: 0.95, alpha: 1.0)
            rect?.layer.cornerRadius = 10
            rect?.layer.masksToBounds = true
            rect?.layer.borderColor = UIColor(named: "Rosa")?.withAlphaComponent(0.2).cgColor
            rect?.layer.borderWidth = 2
            view.sendSubviewToBack(rect!)
        }
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

