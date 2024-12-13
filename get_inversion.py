from typing import Union, Tuple, Optional
import matplotlib.pyplot as plt
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, AutoencoderKL, DDIMScheduler
from torchvision import transforms as tvt
import os
import pickle
from multiprocessing import Process,Value
from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor,CLIPModel
import argparse


@torch.no_grad()
def load_diffmix_embeddings(embed_path: str,
                            text_encoder: CLIPTextModel,
                            tokenizer: CLIPTokenizer,
                            device="cuda",
            ):

    embedding_ckpt = torch.load(embed_path, map_location='cpu')
    learned_embeds_dict = embedding_ckpt["learned_embeds_dict"]
    name2placeholder = embedding_ckpt["name2placeholder"]
    placeholder2name = embedding_ckpt["placeholder2name"]

    name2placeholder = {k.replace('/',' ').replace('_',' '): v for k, v in name2placeholder.items()}
    placeholder2name = {v: k.replace('/',' ').replace('_',' ') for k, v in name2placeholder.items()} 
    
    for token, token_embedding in learned_embeds_dict.items():

        # add the token in tokenizer
        num_added_tokens = tokenizer.add_tokens(token)
        assert num_added_tokens > 0, ERROR_MESSAGE.format(token=token)
    
        # resize the token embeddings
        text_encoder.resize_token_embeddings(len(tokenizer))
        added_token_id = tokenizer.convert_tokens_to_ids(token)

        # get the old word embeddings
        embeddings = text_encoder.get_input_embeddings()

        # get the id for the token and assign new embeds
        embeddings.weight.data[added_token_id] = \
            token_embedding.to(embeddings.weight.dtype)

    return name2placeholder, placeholder2name

def load_image(imgname: str, target_size: Optional[Union[int, Tuple[int, int]]] = None) -> torch.Tensor:
    pil_img = Image.open(imgname).convert('RGB')
    if target_size is not None:
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    return tvt.ToTensor()(pil_img)[None, ...]  # add batch dimension


def img_to_latents(x: torch.Tensor, vae: AutoencoderKL):
    x = 2. * x - 1.
    posterior = vae.encode(x).latent_dist
    latents = posterior.mean * 0.18215
    return latents

def mkdir(path):

    folder = os.path.exists(path)

    if not folder:                   
        os.makedirs(path)  

@torch.no_grad()
def get_ddim_inversion_for_one_category(datasets, shot, category, model_id, inversion_step, condiction_scale, source_dir, des_dir, device):
    if 'imb' in datasets:
        mkdir(des_dir+datasets+'/'+category)
    else:
        mkdir(des_dir+datasets+'/'+shot+'/'+category)
    dtype = torch.float16
    inverse_scheduler = DDIMInverseScheduler.from_pretrained(model_id, subfolder='scheduler')
    pipe = StableDiffusionPipeline.from_pretrained(model_id,
                                                   scheduler=inverse_scheduler,
                                                   safety_checker=None,
                                                   torch_dtype=dtype)
    pipe.to(device)
    vae = pipe.vae
    if 'imb' in datasets:
        pipe.load_lora_weights('ft_ti_db/'+datasets+'/pytorch_lora_weights.safetensors')
        name2placeholder, placeholder2name = load_diffmix_embeddings('ft_ti_db/'+datasets+'/learned_embeds-steps-35000.bin', pipe.text_encoder, pipe.tokenizer)
    else:
        pipe.load_lora_weights('ft_ti_db/'+datasets+'/'+shot+'/pytorch_lora_weights.safetensors')
        name2placeholder, placeholder2name = load_diffmix_embeddings('ft_ti_db/'+datasets+'/'+shot+'/learned_embeds-steps-35000.bin', pipe.text_encoder, pipe.tokenizer)
    place_holder = name2placeholder[category.replace('/',' ').replace('_',' ')]
    if 'cub' in datasets:
        text =  "a photo of a " + place_holder + " bird"
    elif 'aircraft'in datasets:
        text =  "a photo of a " + place_holder + " aircraft"
    elif 'car' in datasets:
        text =  "a photo of a " + place_holder + " car"
    elif 'pet' in datasets:
        text =  "a photo of a " + place_holder + " animal"
    elif 'flower' in datasets:
        text =  "a photo of a " + place_holder + " flower"
    else:
        print("Wrong dataset name!")


    if 'imb' in datasets:
        files = os.listdir(source_dir+datasets+'/train/'+category)
        for file in files:
            context_id = file.split('.')[0]
            input_img = load_image(source_dir+datasets+'/train/'+category+'/'+file, 512).to(device=device, dtype=dtype)


            latents = img_to_latents(input_img, vae)
            inv_latents, _ = pipe(prompt=text, negative_prompt="", guidance_scale=condiction_scale,
                            width=input_img.shape[-1], height=input_img.shape[-2],
                            output_type='latent', return_dict=False,
                            num_inference_steps=inversion_step, latents=latents)

            inv_latens = inv_latents.cpu()
            fw = open(des_dir+datasets+'/'+category+'/'+context_id+'.pkl','wb')
            pickle.dump(inv_latens, fw)
            fw.close()
    else:
        files = os.listdir(source_dir+datasets+'/'+shot+'/train/'+category)
        for file in files:
            context_id = file.split('.')[0]
            input_img = load_image(source_dir+datasets+'/'+shot+'/train/'+category+'/'+file, 512).to(device=device, dtype=dtype)


            latents = img_to_latents(input_img, vae)
            inv_latents, _ = pipe(prompt=text, negative_prompt="", guidance_scale=condiction_scale,
                            width=input_img.shape[-1], height=input_img.shape[-2],
                            output_type='latent', return_dict=False,
                            num_inference_steps=inversion_step, latents=latents)

            inv_latens = inv_latents.cpu()
            fw = open(des_dir+datasets+'/'+shot+'/'+category+'/'+context_id+'.pkl','wb')
            pickle.dump(inv_latens, fw)
            fw.close()
    
    return True

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--datasets',
                    type=str,
                    default='aircraft')
parser.add_argument('--model_id',
                    type=str,
                    default='runwayml/stable-diffusion-v1-5')
parser.add_argument('--inversion_step',
                    type=int,
                    default=200)
parser.add_argument('--condiction_scale',
                    default=1.,
                    type=float)
parser.add_argument('--des_dir',
                    type=str,
                    default='inversions/')
parser.add_argument('--source_dir',
                    type=str,
                    default='datasets/')
parser.add_argument('--shot',
                    type=str,
                    default='5shot')
parser.add_argument('--n_workers',
                    type=int,
                    default=8)

def main():
    args = parser.parse_args()
    if 'cub' in args.datasets:
        class_names = ['Vesper_Sparrow', 'Gadwall', 'Fox_Sparrow', 'Bank_Swallow', 'European_Goldfinch', 'White_throated_Sparrow', 'Hooded_Warbler', 'Baltimore_Oriole', 'White_Pelican', 'Whip_poor_Will', 'Pelagic_Cormorant', 'Prothonotary_Warbler', 'American_Crow', 'Scott_Oriole', 'Scissor_tailed_Flycatcher', 'Gray_Kingbird', 'Clark_Nutcracker', 'Nashville_Warbler', 'Canada_Warbler', 'Cape_Glossy_Starling', 'Evening_Grosbeak', 'White_eyed_Vireo', 'Caspian_Tern', 'Red_legged_Kittiwake', 'Brandt_Cormorant', 'Horned_Grebe', 'Great_Grey_Shrike', 'Ringed_Kingfisher', 'Winter_Wren', 'Pileated_Woodpecker', 'Bobolink', 'Brown_Creeper', 'Brown_Thrasher', 'Tropical_Kingbird', 'Least_Tern', 'Prairie_Warbler', 'Northern_Fulmar', 'Cerulean_Warbler', 'Least_Auklet', 'Geococcyx', 'Sooty_Albatross', 'Ruby_throated_Hummingbird', 'American_Redstart', 'Glaucous_winged_Gull', 'Olive_sided_Flycatcher', 'Common_Tern', 'Magnolia_Warbler', 'Rock_Wren', 'Eastern_Towhee', 'Rhinoceros_Auklet', 'Eared_Grebe', 'Philadelphia_Vireo', 'Cliff_Swallow', 'Seaside_Sparrow', 'Orchard_Oriole', 'Pine_Grosbeak', 'Black_footed_Albatross', 'Red_breasted_Merganser', 'Blue_winged_Warbler', 'Green_tailed_Towhee', 'Vermilion_Flycatcher', 'Mangrove_Cuckoo', 'Nighthawk', 'Red_faced_Cormorant', 'Anna_Hummingbird', 'Western_Meadowlark', 'Red_winged_Blackbird', 'Marsh_Wren', 'Warbling_Vireo', 'California_Gull', 'Yellow_Warbler', 'Gray_Catbird', 'Painted_Bunting', 'Tree_Swallow', 'Ivory_Gull', 'Bay_breasted_Warbler', 'Parakeet_Auklet', 'Blue_Grosbeak', 'Western_Wood_Pewee', 'Savannah_Sparrow', 'Artic_Tern', 'Black_Tern', 'Horned_Puffin', 'Laysan_Albatross', 'Cardinal', 'White_breasted_Kingfisher', 'Carolina_Wren', 'American_Goldfinch', 'Louisiana_Waterthrush', 'Chuck_will_Widow', 'Henslow_Sparrow', 'Pied_billed_Grebe', 'Long_tailed_Jaeger', 'Cactus_Wren', 'Yellow_throated_Vireo', 'Barn_Swallow', 'Sage_Thrasher', 'Mallard', 'Great_Crested_Flycatcher', 'Boat_tailed_Grackle', 'Common_Yellowthroat', 'Forsters_Tern', 'Lincoln_Sparrow', 'American_Pipit', 'Groove_billed_Ani', 'Spotted_Catbird', 'Least_Flycatcher', 'Cape_May_Warbler', 'Pine_Warbler', 'Mockingbird', 'Rusty_Blackbird', 'Field_Sparrow', 'Rufous_Hummingbird', 'Chestnut_sided_Warbler', 'Downy_Woodpecker', 'Clay_colored_Sparrow', 'Gray_crowned_Rosy_Finch', 'Bohemian_Waxwing', 'Le_Conte_Sparrow', 'Black_throated_Sparrow', 'White_crowned_Sparrow', 'Yellow_headed_Blackbird', 'Brewer_Sparrow', 'Harris_Sparrow', 'Sayornis', 'Herring_Gull', 'Loggerhead_Shrike', 'Western_Gull', 'Crested_Auklet', 'Rose_breasted_Grosbeak', 'Lazuli_Bunting', 'Black_throated_Blue_Warbler', 'Red_cockaded_Woodpecker', 'Horned_Lark', 'Blue_headed_Vireo', 'Green_Jay', 'Black_capped_Vireo', 'Red_headed_Woodpecker', 'Ring_billed_Gull', 'Golden_winged_Warbler', 'Frigatebird', 'Green_Kingfisher', 'Chipping_Sparrow', 'Blue_Jay', 'Slaty_backed_Gull', 'Tennessee_Warbler', 'Cedar_Waxwing', 'Belted_Kingfisher', 'Brewer_Blackbird', 'Grasshopper_Sparrow', 'Northern_Waterthrush', 'Bronzed_Cowbird', 'Red_bellied_Woodpecker', 'Hooded_Merganser', 'Worm_eating_Warbler', 'Myrtle_Warbler', 'Pigeon_Guillemot', 'Northern_Flicker', 'American_Three_toed_Woodpecker', 'Indigo_Bunting', 'Green_Violetear', 'Elegant_Tern', 'Red_eyed_Vireo', 'Baird_Sparrow', 'Acadian_Flycatcher', 'Tree_Sparrow', 'Bewick_Wren', 'Pacific_Loon', 'Mourning_Warbler', 'Pomarine_Jaeger', 'Pied_Kingfisher', 'Heermann_Gull', 'Song_Sparrow', 'Western_Grebe', 'House_Wren', 'White_breasted_Nuthatch', 'Dark_eyed_Junco', 'Black_and_white_Warbler', 'Yellow_billed_Cuckoo', 'House_Sparrow', 'Yellow_breasted_Chat', 'Yellow_bellied_Flycatcher', 'Florida_Jay', 'Brown_Pelican', 'Summer_Tanager', 'Orange_crowned_Warbler', 'Ovenbird', 'Purple_Finch', 'Kentucky_Warbler', 'Palm_Warbler', 'Common_Raven', 'Fish_Crow', 'Scarlet_Tanager', 'Hooded_Oriole', 'White_necked_Raven', 'Swainson_Warbler', 'Shiny_Cowbird', 'Nelson_Sharp_tailed_Sparrow', 'Black_billed_Cuckoo', 'Wilson_Warbler']
    elif 'aircraft' in args.datasets:
        class_names = ['707-320', '727-200', '737-200', '737-300', '737-400', '737-500', '737-600', '737-700', '737-800', '737-900', '747-100', '747-200', '747-300', '747-400', '757-200', '757-300', '767-200', '767-300', '767-400', '777-200', '777-300', 'A300B4', 'A310', 'A318', 'A319', 'A320', 'A321', 'A330-200', 'A330-300', 'A340-200', 'A340-300', 'A340-500', 'A340-600', 'A380', 'ATR-42', 'ATR-72', 'An-12', 'BAE_146-200', 'BAE_146-300', 'BAE-125', 'Beechcraft_1900', 'Boeing_717', 'C-130', 'C-47', 'CRJ-200', 'CRJ-700', 'CRJ-900', 'Cessna_172', 'Cessna_208', 'Cessna_525', 'Cessna_560', 'Challenger_600', 'DC-10', 'DC-3', 'DC-6', 'DC-8', 'DC-9-30', 'DH-82', 'DHC-1', 'DHC-6', 'DHC-8-100', 'DHC-8-300', 'DR-400', 'Dornier_328', 'E-170', 'E-190', 'E-195', 'EMB-120', 'ERJ_135', 'ERJ_145', 'Embraer_Legacy_600', 'Eurofighter_Typhoon', 'F-16A_B', 'F_A-18', 'Falcon_2000', 'Falcon_900', 'Fokker_100', 'Fokker_50', 'Fokker_70', 'Global_Express', 'Gulfstream_IV', 'Gulfstream_V', 'Hawk_T1', 'Il-76', 'L-1011', 'MD-11', 'MD-80', 'MD-87', 'MD-90', 'Metroliner', 'Model_B200', 'PA-28', 'SR-20', 'Saab_2000', 'Saab_340', 'Spitfire', 'Tornado', 'Tu-134', 'Tu-154', 'Yak-42']
    elif 'car' in args.datasets:
        class_names = ['am_general_hummer_suv_2000', 'acura_rl_sedan_2012', 'acura_tl_sedan_2012', 'acura_tl_type-s_2008', 'acura_tsx_sedan_2012', 'acura_integra_type_r_2001', 'acura_zdx_hatchback_2012', 'aston_martin_v8_vantage_convertible_2012', 'aston_martin_v8_vantage_coupe_2012', 'aston_martin_virage_convertible_2012', 'aston_martin_virage_coupe_2012', 'audi_rs_4_convertible_2008', 'audi_a5_coupe_2012', 'audi_tts_coupe_2012', 'audi_r8_coupe_2012', 'audi_v8_sedan_1994', 'audi_100_sedan_1994', 'audi_100_wagon_1994', 'audi_tt_hatchback_2011', 'audi_s6_sedan_2011', 'audi_s5_convertible_2012', 'audi_s5_coupe_2012', 'audi_s4_sedan_2012', 'audi_s4_sedan_2007', 'audi_tt_rs_coupe_2012', 'bmw_activehybrid_5_sedan_2012', 'bmw_1_series_convertible_2012', 'bmw_1_series_coupe_2012', 'bmw_3_series_sedan_2012', 'bmw_3_series_wagon_2012', 'bmw_6_series_convertible_2007', 'bmw_x5_suv_2007', 'bmw_x6_suv_2012', 'bmw_m3_coupe_2012', 'bmw_m5_sedan_2010', 'bmw_m6_convertible_2010', 'bmw_x3_suv_2012', 'bmw_z4_convertible_2012', 'bentley_continental_supersports_conv._convertible_2012', 'bentley_arnage_sedan_2009', 'bentley_mulsanne_sedan_2011', 'bentley_continental_gt_coupe_2012', 'bentley_continental_gt_coupe_2007', 'bentley_continental_flying_spur_sedan_2007', 'bugatti_veyron_16.4_convertible_2009', 'bugatti_veyron_16.4_coupe_2009', 'buick_regal_gs_2012', 'buick_rainier_suv_2007', 'buick_verano_sedan_2012', 'buick_enclave_suv_2012', 'cadillac_cts-v_sedan_2012', 'cadillac_srx_suv_2012', 'cadillac_escalade_ext_crew_cab_2007', 'chevrolet_silverado_1500_hybrid_crew_cab_2012', 'chevrolet_corvette_convertible_2012', 'chevrolet_corvette_zr1_2012', 'chevrolet_corvette_ron_fellows_edition_z06_2007', 'chevrolet_traverse_suv_2012', 'chevrolet_camaro_convertible_2012', 'chevrolet_hhr_ss_2010', 'chevrolet_impala_sedan_2007', 'chevrolet_tahoe_hybrid_suv_2012', 'chevrolet_sonic_sedan_2012', 'chevrolet_express_cargo_van_2007', 'chevrolet_avalanche_crew_cab_2012', 'chevrolet_cobalt_ss_2010', 'chevrolet_malibu_hybrid_sedan_2010', 'chevrolet_trailblazer_ss_2009', 'chevrolet_silverado_2500hd_regular_cab_2012', 'chevrolet_silverado_1500_classic_extended_cab_2007', 'chevrolet_express_van_2007', 'chevrolet_monte_carlo_coupe_2007', 'chevrolet_malibu_sedan_2007', 'chevrolet_silverado_1500_extended_cab_2012', 'chevrolet_silverado_1500_regular_cab_2012', 'chrysler_aspen_suv_2009', 'chrysler_sebring_convertible_2010', 'chrysler_town_and_country_minivan_2012', 'chrysler_300_srt-8_2010', 'chrysler_crossfire_convertible_2008', 'chrysler_pt_cruiser_convertible_2008', 'daewoo_nubira_wagon_2002', 'dodge_caliber_wagon_2012', 'dodge_caliber_wagon_2007', 'dodge_caravan_minivan_1997', 'dodge_ram_pickup_3500_crew_cab_2010', 'dodge_ram_pickup_3500_quad_cab_2009', 'dodge_sprinter_cargo_van_2009', 'dodge_journey_suv_2012', 'dodge_dakota_crew_cab_2010', 'dodge_dakota_club_cab_2007', 'dodge_magnum_wagon_2008', 'dodge_challenger_srt8_2011', 'dodge_durango_suv_2012', 'dodge_durango_suv_2007', 'dodge_charger_sedan_2012', 'dodge_charger_srt-8_2009', 'eagle_talon_hatchback_1998', 'fiat_500_abarth_2012', 'fiat_500_convertible_2012', 'ferrari_ff_coupe_2012', 'ferrari_california_convertible_2012', 'ferrari_458_italia_convertible_2012', 'ferrari_458_italia_coupe_2012', 'fisker_karma_sedan_2012', 'ford_f-450_super_duty_crew_cab_2012', 'ford_mustang_convertible_2007', 'ford_freestar_minivan_2007', 'ford_expedition_el_suv_2009', 'ford_edge_suv_2012', 'ford_ranger_supercab_2011', 'ford_gt_coupe_2006', 'ford_f-150_regular_cab_2012', 'ford_f-150_regular_cab_2007', 'ford_focus_sedan_2007', 'ford_e-series_wagon_van_2012', 'ford_fiesta_sedan_2012', 'gmc_terrain_suv_2012', 'gmc_savana_van_2012', 'gmc_yukon_hybrid_suv_2012', 'gmc_acadia_suv_2012', 'gmc_canyon_extended_cab_2012', 'geo_metro_convertible_1993', 'hummer_h3t_crew_cab_2010', 'hummer_h2_sut_crew_cab_2009', 'honda_odyssey_minivan_2012', 'honda_odyssey_minivan_2007', 'honda_accord_coupe_2012', 'honda_accord_sedan_2012', 'hyundai_veloster_hatchback_2012', 'hyundai_santa_fe_suv_2012', 'hyundai_tucson_suv_2012', 'hyundai_veracruz_suv_2012', 'hyundai_sonata_hybrid_sedan_2012', 'hyundai_elantra_sedan_2007', 'hyundai_accent_sedan_2012', 'hyundai_genesis_sedan_2012', 'hyundai_sonata_sedan_2012', 'hyundai_elantra_touring_hatchback_2012', 'hyundai_azera_sedan_2012', 'infiniti_g_coupe_ipl_2012', 'infiniti_qx56_suv_2011', 'isuzu_ascender_suv_2008', 'jaguar_xk_xkr_2012', 'jeep_patriot_suv_2012', 'jeep_wrangler_suv_2012', 'jeep_liberty_suv_2012', 'jeep_grand_cherokee_suv_2012', 'jeep_compass_suv_2012', 'lamborghini_reventon_coupe_2008', 'lamborghini_aventador_coupe_2012', 'lamborghini_gallardo_lp_570-4_superleggera_2012', 'lamborghini_diablo_coupe_2001', 'land_rover_range_rover_suv_2012', 'land_rover_lr2_suv_2012', 'lincoln_town_car_sedan_2011', 'mini_cooper_roadster_convertible_2012', 'maybach_landaulet_convertible_2012', 'mazda_tribute_suv_2011', 'mclaren_mp4-12c_coupe_2012', 'mercedes-benz_300-class_convertible_1993', 'mercedes-benz_c-class_sedan_2012', 'mercedes-benz_sl-class_coupe_2009', 'mercedes-benz_e-class_sedan_2012', 'mercedes-benz_s-class_sedan_2012', 'mercedes-benz_sprinter_van_2012', 'mitsubishi_lancer_sedan_2012', 'nissan_leaf_hatchback_2012', 'nissan_nv_passenger_van_2012', 'nissan_juke_hatchback_2012', 'nissan_240sx_coupe_1998', 'plymouth_neon_coupe_1999', 'porsche_panamera_sedan_2012', 'ram_c_v_cargo_van_minivan_2012', 'rolls-royce_phantom_drophead_coupe_convertible_2012', 'rolls-royce_ghost_sedan_2012', 'rolls-royce_phantom_sedan_2012', 'scion_xd_hatchback_2012', 'spyker_c8_convertible_2009', 'spyker_c8_coupe_2009', 'suzuki_aerio_sedan_2007', 'suzuki_kizashi_sedan_2012', 'suzuki_sx4_hatchback_2012', 'suzuki_sx4_sedan_2012', 'tesla_model_s_sedan_2012', 'toyota_sequoia_suv_2012', 'toyota_camry_sedan_2012', 'toyota_corolla_sedan_2012', 'toyota_4runner_suv_2012', 'volkswagen_golf_hatchback_2012', 'volkswagen_golf_hatchback_1991', 'volkswagen_beetle_hatchback_2012', 'volvo_c30_hatchback_2012', 'volvo_240_sedan_1993', 'volvo_xc90_suv_2007', 'smart_fortwo_convertible_2012']
    elif 'pet' in args.datasets:
        class_names = ['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'boxer', 'chihuahua', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger', 'miniature_pinscher', 'newfoundland', 'pomeranian', 'pug', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier']
    elif 'flower' in args.datasets:
        class_names = ['pink_primrose', 'hard-leaved_pocket_orchid', 'canterbury_bells', 'sweet_pea', 'english_marigold', 'tiger_lily', 'moon_orchid', 'bird_of_paradise', 'monkshood', 'globe_thistle', 'snapdragon', "colt's_foot", 'king_protea', 'spear_thistle', 'yellow_iris', 'globe-flower', 'purple_coneflower', 'peruvian_lily', 'balloon_flower', 'giant_white_arum_lily', 'fire_lily', 'pincushion_flower', 'fritillary', 'red_ginger', 'grape_hyacinth', 'corn_poppy', 'prince_of_wales_feathers', 'stemless_gentian', 'artichoke', 'sweet_william', 'carnation', 'garden_phlox', 'love_in_the_mist', 'mexican_aster', 'alpine_sea_holly', 'ruby-lipped_cattleya', 'cape_flower', 'great_masterwort', 'siam_tulip', 'lenten_rose', 'barbeton_daisy', 'daffodil', 'sword_lily', 'poinsettia', 'bolero_deep_blue', 'wallflower', 'marigold', 'buttercup', 'oxeye_daisy', 'common_dandelion', 'petunia', 'wild_pansy', 'primula', 'sunflower', 'pelargonium', 'bishop_of_llandaff', 'gaura', 'geranium', 'orange_dahlia', 'pink-yellow_dahlia', 'cautleya_spicata', 'japanese_anemone', 'black-eyed_susan', 'silverbush', 'californian_poppy', 'osteospermum', 'spring_crocus', 'bearded_iris', 'windflower', 'tree_poppy', 'gazania', 'azalea', 'water_lily', 'rose', 'thorn_apple', 'morning_glory', 'passion_flower', 'lotus', 'toad_lily', 'anthurium', 'frangipani', 'clematis', 'hibiscus', 'columbine', 'desert-rose', 'tree_mallow', 'magnolia', 'cyclamen_', 'watercress', 'canna_lily', 'hippeastrum_', 'bee_balm', 'ball_moss', 'foxglove', 'bougainvillea', 'camellia', 'mallow', 'mexican_petunia', 'bromelia', 'blanket_flower', 'trumpet_creeper', 'blackberry_lily']
    else:
        print("Wrong dataset name!")
    n_runs = (len(class_names) // args.n_workers) + 1
        
    for i in range(n_runs):
        process_list = []
        if i != (n_runs-1):
            for j in range(args.n_workers):
                p = Process(target=get_ddim_inversion_for_one_category,args=(args.datasets, args.shot, class_names[i*args.n_workers+j], args.model_id, args.inversion_step, args.condiction_scale, args.source_dir, args.des_dir,"cuda:"+str(j)))
                p.start()
                process_list.append(p)
            for each in process_list:
              p.join()
        else:
            for j in range(len(class_names)-i*args.n_workers):
                p = Process(target=get_ddim_inversion_for_one_category,args=(args.datasets, args.shot, class_names[i*args.n_workers+j], args.model_id, args.inversion_step, args.condiction_scale, args.source_dir, args.des_dir,"cuda:"+str(j)))
                p.start()
                process_list.append(p)
            for each in process_list:
              p.join()
    
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
