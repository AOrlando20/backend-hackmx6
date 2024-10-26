//
//  CsvData.swift
//  Liverpool_hack
//
//  Created by Alexis Chávez on 26/10/24.
//

import Foundation

struct Product: Codable {
    let sku: String
    let name: String
    let imagesArray: [String]
    
    enum CodingKeys: String, CodingKey {
        case sku = "SKU"
        case name = "Name"
        // Definimos las claves para las imágenes
        case image1 = "Imagen 1"
        case image2 = "Imagen 2"
        case image3 = "Imagen 3"
        case image4 = "Imagen 4"
        case image5 = "Imagen 5"
        case image6 = "Imagen 6"
        case image7 = "Imagen 7"
        case image8 = "Imagen 8"
        case image9 = "Imagen 9"
        case image10 = "Imagen 10"
        case image11 = "Imagen 11"
        case image12 = "Imagen 12"
        case image13 = "Imagen 13"
        case image14 = "Imagen 14"
        case image15 = "Imagen 15"
        case image16 = "Imagen 16"
        case image17 = "Imagen 17"
        case image18 = "Imagen 18"
        case image19 = "Imagen 19"
        case image20 = "Imagen 20"
        case image21 = "Imagen 21"
        case image22 = "Imagen 22"
        case image23 = "Imagen 23"
        case image24 = "Imagen 24"
        case image25 = "Imagen 25"
        case image26 = "Imagen 26"
        case image27 = "Imagen 27"
        case image28 = "Imagen 28"
        case image29 = "Imagen 29"
    }
    
    // Propiedades individuales para cada imagen
    let image1: String?
    let image2: String?
    let image3: String?
    let image4: String?
    let image5: String?
    let image6: String?
    let image7: String?
    let image8: String?
    let image9: String?
    let image10: String?
    let image11: String?
    let image12: String?
    let image13: String?
    let image14: String?
    let image15: String?
    let image16: String?
    let image17: String?
    let image18: String?
    let image19: String?
    let image20: String?
    let image21: String?
    let image22: String?
    let image23: String?
    let image24: String?
    let image25: String?
    let image26: String?
    let image27: String?
    let image28: String?
    let image29: String?
    
    // Inicializador personalizado para manejar la creación del arreglo de imágenes
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        
        sku = try container.decode(String.self, forKey: .sku)
        name = try container.decode(String.self, forKey: .name)
        
        // Decodificamos las imágenes
        image1 = try container.decodeIfPresent(String.self, forKey: .image1)
        image2 = try container.decodeIfPresent(String.self, forKey: .image2)
        image3 = try container.decodeIfPresent(String.self, forKey: .image3)
        image4 = try container.decodeIfPresent(String.self, forKey: .image4)
        image5 = try container.decodeIfPresent(String.self, forKey: .image5)
        image6 = try container.decodeIfPresent(String.self, forKey: .image6)
        image7 = try container.decodeIfPresent(String.self, forKey: .image7)
        image8 = try container.decodeIfPresent(String.self, forKey: .image8)
        image9 = try container.decodeIfPresent(String.self, forKey: .image9)
        image10 = try container.decodeIfPresent(String.self, forKey: .image10)
        image11 = try container.decodeIfPresent(String.self, forKey: .image11)
        image12 = try container.decodeIfPresent(String.self, forKey: .image12)
        image13 = try container.decodeIfPresent(String.self, forKey: .image13)
        image14 = try container.decodeIfPresent(String.self, forKey: .image14)
        image15 = try container.decodeIfPresent(String.self, forKey: .image15)
        image16 = try container.decodeIfPresent(String.self, forKey: .image16)
        image17 = try container.decodeIfPresent(String.self, forKey: .image17)
        image18 = try container.decodeIfPresent(String.self, forKey: .image18)
        image19 = try container.decodeIfPresent(String.self, forKey: .image19)
        image20 = try container.decodeIfPresent(String.self, forKey: .image20)
        image21 = try container.decodeIfPresent(String.self, forKey: .image21)
        image22 = try container.decodeIfPresent(String.self, forKey: .image22)
        image23 = try container.decodeIfPresent(String.self, forKey: .image23)
        image24 = try container.decodeIfPresent(String.self, forKey: .image24)
        image25 = try container.decodeIfPresent(String.self, forKey: .image25)
        image26 = try container.decodeIfPresent(String.self, forKey: .image26)
        image27 = try container.decodeIfPresent(String.self, forKey: .image27)
        image28 = try container.decodeIfPresent(String.self, forKey: .image28)
        image29 = try container.decodeIfPresent(String.self, forKey: .image29)
        
        // Construimos el arreglo de imágenes
        imagesArray = [
            image1,
            image2,
            image3,
            image4,
            image5,
            image6,
            image7,
            image8,
            image9,
            image10,
            image11,
            image12,
            image13,
            image14,
            image15,
            image16,
            image17,
            image18,
            image19,
            image20,
            image21,
            image22,
            image23,
            image24,
            image25,
            image26,
            image27,
            image28,
            image29
        ].compactMap { $0 }  // Filtrar nil
    }
}
